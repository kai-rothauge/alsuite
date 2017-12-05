#include "clustering.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using boost::mpi;

namespace allib {

KMeans::KMeans() : num_centers(2), max_iterations(20), epsilon(1e-4), init_mode("kmeans||"), init_steps(2), seed(10) { }

KMeans::KMeans(uint32_t _num_centers, uint32_t _max_iterations, double _epsilon, std::string _init_mode,
			uint32_t _init_steps, uint64_t _seed) :
		num_centers(_num_centers), max_iterations(_max_iterations), epsilon(_epsilon), init_mode(_init_mode),
		init_steps(_init_steps), seed(_seed) { }

uint32_t KMeans::get_num_centers() {
	return num_centers;
}

KMeans::set_num_centers(uint32_t _num_centers) {
	num_centers = _num_centers;
}

uint32_t KMeans::get_max_iterations() {
	return max_iterations;
}

void KMeans::set_max_iterations(uint32_t _max_iterations) {
	max_iterations = _max_iterations;
}

std::string KMeans::get_init_mode() {
	return init_mode;
}

void KMeans::set_init_mode(std::string _init_mode) {
	init_mode = _init_mode;
}

uint32_t KMeans::get_init_steps() {
	return init_steps;
}

void KMeans::set_init_steps(uint32_t) {
	init_steps = _init_steps;
}

double KMeans::get_epsilon() {
	return epsilon;
}

void KMeans::set_epsilon(double) {
	epsilon = _epsilon;
}

uint64_t KMeans::get_seed() {
	return seed;
}

void KMeans::set_seed(uint64_t) {
	seed = _seed;
}

void KMeans::set_log(log);
		kmeans.set_world(world);
		kmeans.set_peers(peers);

int KMeans::initialize() {
	strcmp(init_mode, "random") ? initialize_random() : initialize_parallel();
}

int KMeans::train() {

}

int KMeans::run() {



}

uint32_t KMeans::update_assignments_and_counts(MatrixXd const & dataMat, MatrixXd const & centers,
    uint32_t * clusterSizes, std::vector<uint32_t> & rowAssignments, double & objVal) {

	uint32_t num_centers = centers.rows();
	VectorXd distanceSq(num_centers);
	El::Int newAssignment;
	uint32_t numChanged = 0;
	objVal = 0.0;

	for(uint32_t idx = 0; idx < num_centers; ++idx)
		clusterSizes[idx] = 0;

	for(El::Int rowIdx = 0; rowIdx < dataMat.rows(); ++rowIdx) {
		for(uint32_t centerIdx = 0; centerIdx < num_centers; ++centerIdx)
			distanceSq[centerIdx] = (dataMat.row(rowIdx) - centers.row(centerIdx)).squaredNorm();
		objVal += distanceSq.minCoeff(&newAssignment);
		if (rowAssignments[rowIdx] != newAssignment)
			numChanged++;
		rowAssignments[rowIdx] = newAssignment;
		clusterSizes[rowAssignments[rowIdx]] += 1;
	}

	return numChanged;
}

int KMeans::initialize_random {

	return 0;
}

// TODO: add seed as argument (make sure different workers do different things)
int KMeans::initialize_parallel(DistMatrix const * dataMat, MatrixXd const & localData, uint32_t scale, MatrixXd & clusterCenters) {

	auto d = localData.cols();

	// if you have the initial cluster seed, send it to everyone
	uint32_t rowIdx;
	mpi::broadcast(world, rowIdx, 0);
	MatrixXd initialCenter;

	if (dataMat->IsLocalRow(rowIdx)) {
		auto localRowIdx = dataMat->LocalRow(rowIdx);
		initialCenter = localData.row(localRowIdx);
		int maybe_root = 1;
		int rootProcess;
		mpi::all_reduce(peers, peers.rank(), rootProcess, std::plus<int>());
		mpi::broadcast(peers, initialCenter, peers.rank());
	}
	else {
		int maybe_root = 0;
		int rootProcess;
		mpi::all_reduce(peers, 0, rootProcess, std::plus<int>());
		mpi::broadcast(peers, initialCenter, rootProcess);
	}

	//in each step, sample 2*k points on average (totalled across the partitions)
	// with probability proportional to their squared distance from the current
	// cluster centers and add the sampled points to the set of cluster centers
	std::vector<double> distSqToCenters(localData.rows());
	double Z; // normalization constant
	std::mt19937 gen(seed + world.rank());
	std::uniform_real_distribution<double> dis(0, 1);

	std::vector<MatrixXd> initCenters;
	initCenters.push_back(initialCenter);

	for(int steps = 0; steps < initSteps; ++steps) {
		// 1) compute the distance of your points from the set of centers and all_reduce
		// to get the normalization for the sampling probability
		VectorXd distSq(initCenters.size());
		Z = 0;
		for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
			for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
				distSq[centerIdx] = (localData.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
			distSqToCenters[pointIdx] = distSq.minCoeff();
			Z += distSqToCenters[pointIdx];
		}
		mpi::all_reduce(peers, mpi::inplace_t<double>(Z), std::plus<double>());

		// 2) sample your points accordingly
		std::vector<MatrixXd> localNewCenters;
		for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
			bool sampledQ = ( dis(gen) < ((double)scale) * distSqToCenters[pointIdx]/Z ) ? true : false;
			if (sampledQ) {
				localNewCenters.push_back(localData.row(pointIdx));
			}
		}

		// 3) have each worker broadcast out their sampled points to all other workers,
		// to update each worker's set of centers
		for(uint32_t root= 0; root < peers.size(); ++root) {
			if (root == peers.rank()) {
				mpi::broadcast(peers, localNewCenters, root);
				initCenters.insert(initCenters.end(), localNewCenters.begin(), localNewCenters.end());
			} else {
				std::vector<MatrixXd> remoteNewCenters;
				mpi::broadcast(peers, remoteNewCenters, root);
				initCenters.insert(initCenters.end(), remoteNewCenters.begin(), remoteNewCenters.end());
			}
		}
	} // end for

	// figure out the number of points closest to each cluster center
	std::vector<uint32_t> clusterSizes(initCenters.size(), 0);
	std::vector<uint32_t> localClusterSizes(initCenters.size(), 0);
	VectorXd distSq(initCenters.size());
	for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
		for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
			distSq[centerIdx] = (localData.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
		uint32_t clusterIdx;
		distSq.minCoeff(&clusterIdx);
		localClusterSizes[clusterIdx] += 1;
	}

	mpi::all_reduce(peers, localClusterSizes.data(), localClusterSizes.size(),
	clusterSizes.data(), std::plus<uint32_t>());

	// after centers have been sampled, sync back up with the driver,
	// and send them there for local clustering
	world.barrier();
	if (world.rank() == 1) {
		world.send(0, 0, clusterSizes);
		world.send(0, 0, initCenters);
	}
	world.barrier();

	clusterCenters.setZero();
	mpi::broadcast(world, clusterCenters.data(), clusterCenters.rows()*d, 0);

	return 0;
}

int KMeans::kmeansPP(std::vector<MatrixXd> points, std::vector<double> weights, MatrixXd & fitCenters) {

	std::default_random_engine randGen(seed);
	std::uniform_real_distribution<double> unifReal(0.0, 1.0);
	uint32_t n = points.size();
	uint32_t d = points[0].cols();
	uint32_t k = fitCenters.rows();
	std::vector<uint32_t> pointIndices(n);
	std::vector<uint32_t> centerIndices(k);
	std::iota(pointIndices.begin(), pointIndices.end(), 0);
	std::iota(centerIndices.begin(), centerIndices.end(), 0);

	// pick initial cluster center using weighted sampling
	double stopSum = unifReal(randGen)*std::accumulate(weights.begin(), weights.end(), 0.0);
	double curSum = 0.0;
	uint32_t searchIdx = 0;
	while(searchIdx < n && curSum < stopSum) {
		curSum += weights[searchIdx];
		searchIdx += 1;
	}
	fitCenters.row(0) = points[searchIdx - 1];

	// iteratively select next cluster centers with
	// probability proportional to the squared distance from the previous centers
	// recall we are doing weighted k-means so min sum(w_i*d(x_i,C)) rather than sum(d(x_i,C))


	auto start = std::chrono::system_clock::now();
	std::vector<double> samplingDensity(n);
	for(auto pointIdx : pointIndices) {
		samplingDensity[pointIdx] = weights[pointIdx]*(points[pointIdx] - fitCenters.row(0)).squaredNorm();
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double, std::milli> elapsed_ms(end - start);

	for(uint32_t centerSelectionIdx = 1; centerSelectionIdx < k; centerSelectionIdx++) {
		stopSum = unifReal(randGen)*std::accumulate(samplingDensity.begin(), samplingDensity.end(), 0.0);
		curSum = 0.0;
		searchIdx = 0;
		while(searchIdx < n && curSum < stopSum) {
			curSum += samplingDensity[searchIdx];
			searchIdx += 1;
		}
		// if less than k initial points explain all the data, set remaining centers to the first point
		fitCenters.row(centerSelectionIdx) = searchIdx > 0 ? points[searchIdx - 1] : points[0];
		for(auto pointIdx : pointIndices)
			samplingDensity[pointIdx] = std::min(samplingDensity[pointIdx],
					weights[pointIdx]*(points[pointIdx] - fitCenters.row(centerSelectionIdx)).squaredNorm());

		//    std::cerr << Eigen::Map<Eigen::RowVectorXd>(samplingDensity.data(), n) << std::endl;
	}

	// run Lloyd's algorithm stop when reached max iterations or points stop changing cluster assignments
	bool movedQ;
	std::vector<double> clusterSizes(k, 0.0);
	std::vector<uint32_t> clusterAssignments(n, 0);
	MatrixXd clusterPointSums(k, d);
	VectorXd distanceSqToCenters(k);
	uint32_t newClusterAssignment;
	double sqDist;

	uint32_t iter = 0;
	for(; iter < maxIters; iter++) {
		movedQ = false;
		clusterPointSums.setZero();
		std::fill(clusterSizes.begin(), clusterSizes.end(), 0);

		// assign each point to nearest cluster and count number of points in each cluster
		for(auto pointIdx : pointIndices) {
			for(auto centerIdx : centerIndices)
				distanceSqToCenters(centerIdx) = (points[pointIdx] - fitCenters.row(centerIdx)).squaredNorm();
			sqDist = distanceSqToCenters.minCoeff(&newClusterAssignment);
			if (newClusterAssignment != clusterAssignments[pointIdx])
				movedQ = true;
			clusterAssignments[pointIdx] = newClusterAssignment;
			clusterPointSums.row(newClusterAssignment) += weights[pointIdx]*points[pointIdx];
			clusterSizes[newClusterAssignment] += weights[pointIdx];
		}

		// stop iterations if cluster assignments have not changed
		if(!movedQ) break;

		// update cluster centers
		for(auto centerIdx : centerIndices) {
			if ( clusterSizes[centerIdx] > 0 ) {
				fitCenters.row(centerIdx) = clusterPointSums.row(centerIdx) / clusterSizes[centerIdx];
			} else {
				uint32_t randPtIdx = (uint32_t) std::round(unifReal(randGen)*n);
				fitCenters.row(centerIdx) = points[randPtIdx];
			}
		}
	}

	// seems necessary to force eigen to return the centers as an actual usable matrix
	for(uint32_t rowidx = 0; rowidx < k; rowidx++)
		for(uint32_t colidx = 0; colidx < k; colidx++)
			fitCenters(rowidx, colidx) = fitCenters(rowidx, colidx) + 0.0;

	return 0;
}

int KMeans::run() {
	initialize();
	train();
}

int KMeans::train() {

	bool isDriver = world.rank() == 0;
	if isDriver {

		auto log = start_log("AlLib driver");

		log->info("Starting K-means on matrix {}", inputMat);
		log->info("num_centers = {}, max_iterations = {}, epsilon = {}, init_method = {}, init_steps = {}, seed = {}",
		num_centers, max_iterations, epsilon, init_method, init_steps, seed);

		if (epsilon < 0.0 || epsilon > 1.0) {
			log->error("Unreasonable change threshold in k-means: {}", changeThreshold);
			abort();
		}
		if (method != 1) {
			log->warn("Sorry, only k-means|| initialization has been implemented, so ignoring your choice of method {}", method);
		}

		auto n = matrices[inputMat].numRows;
		auto d = matrices[inputMat].numCols;
		MatrixHandle centersHandle = this->registerMatrix(num_centers, d);
		MatrixHandle assignmentsHandle = this->registerMatrix(n, 1);
		KMeansCommand cmd(inputMat, num_centers, method, initSteps, changeThreshold, seed, centersHandle, assignmentsHandle);
		issue(cmd); // initial call initializes stuff and waits for next command

		/******** START of kmeans|| initialization ********/
		std::mt19937 gen(seed);
		std::uniform_int_distribution<unsigned long> dis(0, n-1);
		uint32_t rowidx = dis(gen);
		std::vector<double> initialCenter(d);

		mpi::broadcast(world, rowidx, 0); // tell the workers which row to use as initialization in kmeans||
		world.barrier(); // wait for workers to return oversampled cluster centers and sizes

		std::vector<uint32_t> clusterSizes;
		std::vector<MatrixXd> initClusterCenters;
		world.recv(1, mpi::any_tag, clusterSizes);
		world.recv(1, mpi::any_tag, initClusterCenters);
		world.barrier();

		log->info("Retrieved the k-means|| oversized set of potential cluster centers");
		log->debug("{}", initClusterCenters);

		// use kmeans++ locally to find the initial cluster centers
		std::vector<double> weights;
		weights.reserve(clusterSizes.size());
		std::for_each(clusterSizes.begin(), clusterSizes.end(), [&weights](const uint32_t & cnt){ weights.push_back((double) cnt); });
		MatrixXd clusterCenters(num_centers, d);

		kmeansPP(gen(), initClusterCenters, weights, clusterCenters, 30); // same number of maxIters as spark kmeans

		log->info("Ran local k-means on the driver to determine starting cluster centers");
		log->debug("{}", clusterCenters);

		mpi::broadcast(world, clusterCenters.data(), num_centers*d, 0);
		/******** END of kMeans|| initialization ********/

		/******** START of Lloyd's algorithm iterations ********/
		double percentAssignmentsChanged = 1.0;
		bool centersMovedQ = true;
		uint32_t numChanged = 0;
		uint32_t numIters = 0;
		std::vector<uint32_t> parClusterSizes(num_centers);
		std::vector<uint32_t> zerosVector(num_centers);

		for(uint32_t clusterIdx = 0; clusterIdx < num_centers; clusterIdx++)
		zerosVector[clusterIdx] = 0;

		uint32_t command = 1; // do another iteration
		while (centersMovedQ && numIters++ < max_iterations)  {
			log->info("Starting iteration {} of Lloyd's algorithm, {} percentage changed in last iter",
			numIters, percentAssignmentsChanged*100);
			numChanged = 0;
			for(uint32_t clusterIdx = 0; clusterIdx < num_centers; clusterIdx++)
				parClusterSizes[clusterIdx] = 0;
			command = 1; // do a basic iteration
			mpi::broadcast(world, command, 0);
			mpi::reduce(world, (uint32_t) 0, numChanged, std::plus<int>(), 0);
			mpi::reduce(world, zerosVector.data(), num_centers, parClusterSizes.data(), std::plus<uint32_t>(), 0);
			world.recv(1, mpi::any_tag, centersMovedQ);
			percentAssignmentsChanged = ((double) numChanged)/n;

			for(uint32_t clusterIdx = 0; clusterIdx < num_centers; clusterIdx++) {
				if (parClusterSizes[clusterIdx] == 0) {
					// this is an empty cluster, so randomly pick a point in the dataset
					// as that cluster's centroid
					centersMovedQ = true;
					command = 2; // reinitialize this cluster center
					uint32_t rowIdx = dis(gen);
					mpi::broadcast(world, command, 0);
					mpi::broadcast(world, clusterIdx, 0);
					mpi::broadcast(world, rowIdx, 0);
					world.barrier();
				}
			}
		}
		command = 0xf; // terminate and finalize the k-means centers and assignments as distributed matrices
		mpi::broadcast(world, command, 0);
		double objVal = 0.0;
		mpi::reduce(world, 0.0, objVal, std::plus<double>(), 0);
		world.barrier();

		/******** END of Lloyd's iterations ********/

		log->info("Finished Lloyd's algorithm: took {} iterations, final objective value {}", numIters, objVal);
		output.writeInt(0x1);
		output.writeInt(assignmentsHandle.id);
		output.writeInt(centersHandle.id);
		output.writeInt(numIters);
		output.flush();
		}

	}
	else {

		log->info("Started kmeans");
		auto origDataMat = matrices[origMat].get();
		auto n = origDataMat->Height();
		auto d = origDataMat->Width();

		// relayout matrix if needed so that it is in row-partitioned format
		// cf http://libelemental.org/pub/slides/ICS13.pdf slide 19 for the cost of redistribution
		auto distData = origDataMat->DistData();
		DistMatrix * dataMat = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(n, d, grid);
		if (distData.colDist == El::MD && distData.rowDist == El::STAR) {
			dataMat = origDataMat;
		} else {
			auto relayoutStart = std::chrono::system_clock::now();
			El::Copy(*origDataMat, *dataMat); // relayouts data so it is row-wise partitioned
			std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - relayoutStart);
			log->info("Detected matrix is not row-partitioned, so relayouted to row-partitioned; took {} ms ", relayoutDuration.count());
		}

		// TODO: store these as local matrices on the driver
		DistMatrix * centers = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(num_centers, d, grid);
		DistMatrix * assignments = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(n, 1, grid);
		ENSURE(matrices.insert(std::make_pair(centersHandle, std::unique_ptr<DistMatrix>(centers))).second);
		ENSURE(matrices.insert(std::make_pair(assignmentsHandle, std::unique_ptr<DistMatrix>(assignments))).second);

		MatrixXd localData(dataMat->LocalHeight(), d);

		// compute the map from local row indices to the row indices in the global matrix
		// and populate the local data matrix

		std::vector<El::Int> rowMap(localData.rows());
		for(El::Int rowIdx = 0; rowIdx < n; ++rowIdx)
			if (dataMat->IsLocalRow(rowIdx)) {
				auto localRowIdx = dataMat->LocalRow(rowIdx);
				rowMap[localRowIdx] = rowIdx;
				for(El::Int colIdx = 0; colIdx < d; ++colIdx)
					localData(localRowIdx, colIdx) = dataMat->GetLocal(localRowIdx, colIdx);
			}

		MatrixXd clusterCenters(num_centers, d);
		MatrixXd oldClusterCenters(num_centers, d);

		// initialize centers using kMeans||
		uint32_t scale = 2*num_centers;
		clusterCenters.setZero();
		kmeansParallelInit(self, dataMat, localData, scale, initSteps, clusterCenters, seed);

		// TODO: allow to initialize k-means randomly
		//MatrixXd clusterCenters = MatrixXd::Random(num_centers, d);

		/******** START Lloyd's iterations ********/
		// compute the local cluster assignments
		std::unique_ptr<uint32_t[]> counts{new uint32_t[num_centers]};
		std::vector<uint32_t> rowAssignments(localData.rows());
		VectorXd distanceSq(num_centers);
		double objVal;

		update_assignments_and_counts(localData, clusterCenters, counts.get(), rowAssignments, objVal);

		MatrixXd centersBuf(num_centers, d);
		std::unique_ptr<uint32_t[]> countsBuf{new uint32_t[num_centers]};
		uint32_t numChanged = 0;
		oldClusterCenters = clusterCenters;

		while(true) {
			uint32_t nextCommand;
			mpi::broadcast(world, nextCommand, 0);

			if (nextCommand == 0xf)  // finished iterating
				break;
			else if (nextCommand == 2) { // encountered an empty cluster, so randomly pick a point in the dataset as that cluster's centroid
				uint32_t clusterIdx, rowIdx;
				mpi::broadcast(world, clusterIdx, 0);
				mpi::broadcast(world, rowIdx, 0);
				if (dataMat->IsLocalRow(rowIdx)) {
					auto localRowIdx = dataMat->LocalRow(rowIdx);
					clusterCenters.row(clusterIdx) = localData.row(localRowIdx);
				}
				mpi::broadcast(peers, clusterCenters, peers.rank());
				updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
				world.barrier();
				continue;
			}

			/******** do a regular Lloyd's iteration ********/

			// update the centers
			// TODO: locally compute cluster sums and place in clusterCenters
			oldClusterCenters = clusterCenters;
			clusterCenters.setZero();
			for(uint32_t rowIdx = 0; rowIdx < localData.rows(); ++rowIdx)
				clusterCenters.row(rowAssignments[rowIdx]) += localData.row(rowIdx);

			mpi::all_reduce(peers, clusterCenters.data(), num_centers*d, centersBuf.data(), std::plus<double>());
			std::memcpy(clusterCenters.data(), centersBuf.data(), num_centers*d*sizeof(double));
			mpi::all_reduce(peers, counts.get(), num_centers, countsBuf.get(), std::plus<uint32_t>());
			std::memcpy(counts.get(), countsBuf.get(), num_centers*sizeof(uint32_t));

			for(uint32_t rowIdx = 0; rowIdx < num_centers; ++rowIdx)
				if( counts[rowIdx] > 0)
					clusterCenters.row(rowIdx) /= counts[rowIdx];

			// compute new local assignments
			numChanged = updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
			std::cerr << "computed Updated assingments\n" << std::flush;

			// return the number of changed assignments
			mpi::reduce(world, numChanged, std::plus<int>(), 0);
			// return the cluster counts
			mpi::reduce(world, counts.get(), num_centers, std::plus<uint32_t>(), 0);
			std::cerr << "returned cluster counts\n" << std::flush;
			if (world.rank() == 1) {
				bool movedQ = (clusterCenters - oldClusterCenters).rowwise().norm().minCoeff() > changeThreshold;
				world.send(0, 0, movedQ);
			}
		}

		// write the final k-means centers and assignments
		auto startKMeansWrite = std::chrono::system_clock::now();
		El::Zero(*assignments);
		assignments->Reserve(localData.rows());
		for(El::Int rowIdx = 0; rowIdx < localData.rows(); ++rowIdx)
			assignments->QueueUpdate(rowMap[rowIdx], 0, rowAssignments[rowIdx]);
		assignments->ProcessQueues();

		El::Zero(*centers);
		centers->Reserve(centers->LocalHeight()*d);
		for(uint32_t clusterIdx = 0; clusterIdx < num_centers; ++clusterIdx)
			if (centers->IsLocalRow(clusterIdx)) {
				for(El::Int colIdx = 0; colIdx < d; ++colIdx)
					centers->QueueUpdate(clusterIdx, colIdx, clusterCenters(clusterIdx, colIdx));
			}
		centers->ProcessQueues();
		std::chrono::duration<double, std::milli> kMeansWrite_duration(std::chrono::system_clock::now() - startKMeansWrite);
		std::cerr << world.rank() << ": writing the k-means centers and assignments took " << kMeansWrite_duration.count() << "ms\n";

		mpi::reduce(world, objVal, std::plus<double>(), 0);
		world.barrier();
	}

	output.add(new StringParameter("result", "success"));
	output.add(new FloatParameter("error", 5.555555f));

	return 0;
}

} // namespace allib
