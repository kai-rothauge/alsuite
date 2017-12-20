#include "SVD.hpp"

namespace allib {

int SVD::get_rank() {
	return rank;
}

void SVD::set_rank(uint64_t _rank) {
	rank = _rank;
}

void SVD::set_data_matrix(DistMatrix * _A) {
	A = _A;
}

int SVD::run(Parameters & output) {

	uint32_t command = 0;

	if (world.rank() == 0) {

		auto n = A->Height();
		auto d = A->Width();

		log->info("Starting truncated SVD on {}x{} matrix", n, d);
		log->info("Settings:");
		log->info("    rank = {}", rank);

		boost::mpi::broadcast(world, command, 0);		// Tell workers to start

//		log->info("Starting truncated SVD computation");
//		MatrixHandle inputMat{input.readInt()};
//		uint32_t k = input.readInt();

//		MatrixHandle UHandle{nextMatrixId++};
//		MatrixHandle SHandle{nextMatrixId++};
//		MatrixHandle VHandle{nextMatrixId++};

//		auto m = matrices[inputMat].numRows;
//		auto n = matrices[inputMat].numCols;
//		TruncatedSVDCommand cmd(inputMat, UHandle, SHandle, VHandle, k);
//		issue(cmd);

		ARrcSymStdEig<double> prob(n, rank, "LM");
		std::vector<double> zerosVector(n);
		for (uint32_t idx = 0; idx < n; idx++)
			zerosVector[idx] = 0;

		int iterNum = 0;

		while (!prob.ArnoldiBasisFound()) {
			prob.TakeStep();
			++iterNum;
			if (iterNum % 20 == 0) {
				log->info("Computed {} mv products", iterNum);
			}
			if (prob.GetIdo() == 1 || prob.GetIdo() == -1) {
				command = 1;
				boost::mpi::broadcast(world, command, 0);
				boost::mpi::broadcast(world, prob.GetVector(), n, 0);
				boost::mpi::reduce(world, zerosVector.data(), n, prob.PutVector(), std::plus<double>(), 0);
			}
		}

		prob.FindEigenvectors();
		uint32_t nconv = prob.ConvergedEigenvalues();
		uint32_t niters = prob.GetIter();
		log->info("Done after {} Arnoldi iterations, converged to {} eigenvectors of size {}", niters, nconv, n);

		//NB: it may be the case that n*nconv > 4 GB, then have to be careful!
		// assuming tall and skinny A for now
		MatrixXd rightVecs(n, nconv);
		log->info("Allocated matrix for right eivenvectors of A'*A");
		// Eigen uses column-major layout by default!
		for(uint32_t idx = 0; idx < nconv; idx++)
			std::memcpy(rightVecs.col(idx).data(), prob.RawEigenvector(idx), n*sizeof(double));
		log->info("Copied right eigenvectors into allocated storage");

		// Populate U, V, S
		command = 2;
		boost::mpi::broadcast(world, command, 0);
		boost::mpi::broadcast(world, nconv, 0);
		log->info("Broadcasted command and number of converged eigenvectors");
		boost::mpi::broadcast(world, rightVecs.data(), n*nconv, 0);
		log->info("Broadcasted right eigenvectors");
		boost::mpi::broadcast(world, prob.RawEigenvalues(), nconv, 0);
		log->info("Broadcasted eigenvalues");

//		MatrixDescriptor Uinfo(UHandle, m, nconv);
//		MatrixDescriptor Sinfo(SHandle, nconv, 1);
//		MatrixDescriptor Vinfo(VHandle, n, nconv);
//		ENSURE(matrices.insert(std::make_pair(UHandle, Uinfo)).second);
//		ENSURE(matrices.insert(std::make_pair(SHandle, Sinfo)).second);
//		ENSURE(matrices.insert(std::make_pair(VHandle, Vinfo)).second);

		world.barrier();
	}
	else {

		int command;
		boost::mpi::broadcast(world, command, 0);

		if (command == 0) {
			log->info("Started truncated SVD");

			auto m = A->Height();
			auto n = A->Width();

			// Relayout matrix so it is row-partitioned
			DistMatrix * workingMat;
			std::unique_ptr<DistMatrix> dataMat_uniqptr{new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(m, n, grid)}; // so will delete this relayed out matrix once kmeans goes out of scope
			auto distData = A->DistData();
			if (distData.colDist == El::MD && distData.rowDist == El::STAR) {
				workingMat = A;
			}
			else {
				log->info("detected matrix is not row-partitioned, so relayout-ing a copy to row-partitioned");
				workingMat = dataMat_uniqptr.get();
				auto relayoutStart = std::chrono::system_clock::now();
				El::Copy(*A, *workingMat); // relayouts data so it is row-wise partitioned
				std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - relayoutStart);
				log->info("relayout took {} ms", relayoutDuration.count());
			}

			// Retrieve your local block of rows and compute local contribution to A'*A
			MatrixXd localData(workingMat->LocalHeight(), n);

			log->info("Computing the local contribution to A'*A");
			auto startFillLocalMat = std::chrono::system_clock::now();

			// may be SUPERSLOW? use LockedBuffers for local manipulations?
			/*
			log->info("Computing local contribution to the Gramian matrix");
			std::vector<El::Int> rowMap(localData.rows());
			for(El::Int rowIdx = 0; rowIdx < m; ++rowIdx)
			if (workingMat->IsLocalRow(rowIdx)) {
			  auto localRowIdx = workingMat->LocalRow(rowIdx);
			  rowMap[localRowIdx] = rowIdx;
			  for(El::Int colIdx = 0; colIdx < n; ++colIdx)
				localData(localRowIdx, colIdx) = workingMat->GetLocal(localRowIdx, colIdx);
			}
			*/

			// Need to double-check is doing the right thing
			log->info("Extracting the local rows");
			const El::Matrix<double> &localMat = workingMat->LockedMatrix();
			const double * localChunk = localMat.LockedBuffer();
			for (El::Int rowIdx = 0; rowIdx < localMat.Height(); ++rowIdx)
				for (El::Int colIdx = 0; colIdx < n; ++colIdx)
					localData(rowIdx, colIdx) = localChunk[colIdx * localMat.LDim() + rowIdx];
			log->info("Done extracting the local rows, now computing the local Gramian");
			//log->info("Using {} threads", Eigen::nbThreads());

			// NB: Sometimes it makes sense to precompute the gramMat (when it's cheap (we have a lot of cores and enough memory), sometimes
			// it makes more sense to compute A'*(A*x) separately each time (when we don't have enough memory for gramMat, or its too expensive
			// time-wise to precompute GramMat). Trade-off depends on k (through the number of Arnoldi iterations we'll end up needing), the
			// amount of memory we have free to store GramMat, and the number of cores we have available
			MatrixXd gramMat = localData.transpose()*localData;
			std::chrono::duration<double, std::milli> fillLocalMat_duration(std::chrono::system_clock::now() - startFillLocalMat);
			log->info("Took {} ms to compute local contribution to A'*A", fillLocalMat_duration.count());

			std::unique_ptr<double[]> vecIn{new double[n]};

			log->info("Finished initialization for truncated SVD");

			while (true) {
				boost::mpi::broadcast(world, command, 0);
				if (command == 1) {
					boost::mpi::broadcast(world, vecIn.get(), n, 0);

					Eigen::Map<VectorXd> x(vecIn.get(), n);
					auto startMvProd = std::chrono::system_clock::now();
					VectorXd y = gramMat * x;
					//VectorXd y = localData.transpose()* (localData*x);
					std::chrono::duration<double, std::milli> elapsed_msMvProd(std::chrono::system_clock::now() - startMvProd);
					//std::cerr << world.rank() << ": Took " << elapsed_msMvProd.count() << "ms to multiply A'*A*x\n";

					boost::mpi::reduce(world, y.data(), n, std::plus<double>(), 0);
				}
				if (command == 2) {
					uint32_t nconv;
					boost::mpi::broadcast(world, nconv, 0);

					MatrixXd rightEigs(n, nconv);
					boost::mpi::broadcast(world, rightEigs.data(), n*nconv, 0);
					VectorXd singValsSq(nconv);
					boost::mpi::broadcast(world, singValsSq.data(), nconv, 0);

					DistMatrix * U = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, nconv, grid);
					DistMatrix * S = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(nconv, 1, grid);
					DistMatrix * Sinv = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(nconv, 1, grid);
					DistMatrix * V = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, nconv, grid);

					output.add_distmatrix("U", U);
					output.add_distmatrix("S", S);
					output.add_distmatrix("V", V);

					// populate V
					for (El::Int rowIdx=0; rowIdx < n; rowIdx++)
						for (El::Int colIdx=0; colIdx < (El::Int) nconv; colIdx++)
							if (V->IsLocal(rowIdx, colIdx))
								V->SetLocal(V->LocalRow(rowIdx), V->LocalCol(colIdx), rightEigs(rowIdx,colIdx));
					// populate S, Sinv
					for (El::Int idx=0; idx < (El::Int) nconv; idx++) {
						if (S->IsLocal(idx, 0))
							S->SetLocal(S->LocalRow(idx), 0, std::sqrt(singValsSq(idx)));
						if (Sinv->IsLocal(idx, 0))
							Sinv->SetLocal(Sinv->LocalRow(idx), 0, 1/std::sqrt(singValsSq(idx)));
					}

					// form U
					El::Gemm(El::NORMAL, El::NORMAL, 1.0, *A, *V, 0.0, *U);
					// TODO: do a QR instead, but does column pivoting so would require postprocessing S,V to stay consistent
					El::DiagonalScale(El::RIGHT, El::NORMAL, *Sinv, *U);

					break;
				}
			}

			world.barrier();

		}
	}


	return 0;
}

} // namespace allib
