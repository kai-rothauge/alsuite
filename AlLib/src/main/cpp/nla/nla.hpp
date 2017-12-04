#ifndef ALML_NLA_HPP
#define ALML_NLA_HPP

#include "matrix/matrix.hpp"

#endif // ALML_NLA_HPP


void TruncatedSVDCommand::run(Worker *self) const {
  auto m = self->matrices[mat]->Height();
  auto n = self->matrices[mat]->Width();
  auto A = self->matrices[mat].get();

  // Relayout matrix so it is row-partitioned
  DistMatrix * workingMat;
  std::unique_ptr<DistMatrix> dataMat_uniqptr{new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(m, n, self->grid)}; // so will delete this relayed out matrix once kmeans goes out of scope
  auto distData = A->DistData();
  if (distData.colDist == El::MD && distData.rowDist == El::STAR) {
    workingMat = A;
  } else {
    self->log->info("detected matrix is not row-partitioned, so relayout-ing a copy to row-partitioned");
    workingMat = dataMat_uniqptr.get();
    auto relayoutStart = std::chrono::system_clock::now();
    El::Copy(*A, *workingMat); // relayouts data so it is row-wise partitioned
    std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - relayoutStart);
    self->log->info("relayout took {} ms", relayoutDuration.count());
  }

  // Retrieve your local block of rows and compute local contribution to A'*A
  MatrixXd localData(workingMat->LocalHeight(), n);

  self->log->info("Computing the local contribution to A'*A");
  auto startFillLocalMat = std::chrono::system_clock::now();

  // may be SUPERSLOW? use LockedBuffers for local manipulations?
  /*
  self->log->info("Computing local contribution to the Gramian matrix");
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
  self->log->info("Extracting the local rows");
  const El::Matrix<double> &localMat = workingMat->LockedMatrix();
  const double * localChunk = localMat.LockedBuffer();
  for(El::Int rowIdx = 0; rowIdx < localMat.Height(); ++rowIdx)
      for(El::Int colIdx = 0; colIdx < n; ++colIdx)
          localData(rowIdx, colIdx) = localChunk[colIdx * localMat.LDim() + rowIdx];
  self->log->info("Done extracting the local rows, now computing the local Gramian");
  //self->log->info("Using {} threads", Eigen::nbThreads());

  //NB: sometimes it makes sense to precompute the gramMat (when it's cheap (we have a lot of cores and enough memory), sometimes
  // it makes more sense to compute A'*(A*x) separately each time (when we don't have enough memory for gramMat, or its too expensive
  // time-wise to precompute GramMat). trade-off depends on k (through the number of Arnoldi iterations we'll end up needing), the
  // amount of memory we have free to store GramMat, and the number of cores we have available
  MatrixXd gramMat = localData.transpose()*localData;
  std::chrono::duration<double, std::milli> fillLocalMat_duration(std::chrono::system_clock::now() - startFillLocalMat);
  self->log->info("Took {} ms to compute local contribution to A'*A", fillLocalMat_duration.count());

  uint32_t command;
  std::unique_ptr<double[]> vecIn{new double[n]};

  self->log->info("finished initialization for truncated SVD");

  while(true) {
    mpi::broadcast(self->world, command, 0);
    if (command == 1) {
      mpi::broadcast(self->world, vecIn.get(), n, 0);

      Eigen::Map<VectorXd> x(vecIn.get(), n);
      auto startMvProd = std::chrono::system_clock::now();
      VectorXd y = gramMat * x;
      //VectorXd y = localData.transpose()* (localData*x);
      std::chrono::duration<double, std::milli> elapsed_msMvProd(std::chrono::system_clock::now() - startMvProd);
      //std::cerr << self->world.rank() << ": Took " << elapsed_msMvProd.count() << "ms to multiply A'*A*x\n";

      mpi::reduce(self->world, y.data(), n, std::plus<double>(), 0);
    }
    if (command == 2) {
      uint32_t nconv;
      mpi::broadcast(self->world, nconv, 0);

      MatrixXd rightEigs(n, nconv);
      mpi::broadcast(self->world, rightEigs.data(), n*nconv, 0);
      VectorXd singValsSq(nconv);
      mpi::broadcast(self->world, singValsSq.data(), nconv, 0);

      DistMatrix * U = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, nconv, self->grid);
      DistMatrix * S = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(nconv, 1, self->grid);
      DistMatrix * Sinv = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(nconv, 1, self->grid);
      DistMatrix * V = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, nconv, self->grid);

      ENSURE(self->matrices.insert(std::make_pair(UHandle, std::unique_ptr<DistMatrix>(U))).second);
      ENSURE(self->matrices.insert(std::make_pair(SHandle, std::unique_ptr<DistMatrix>(S))).second);
      ENSURE(self->matrices.insert(std::make_pair(VHandle, std::unique_ptr<DistMatrix>(V))).second);

      // populate V
      for(El::Int rowIdx=0; rowIdx < n; rowIdx++)
        for(El::Int colIdx=0; colIdx < (El::Int) nconv; colIdx++)
          if(V->IsLocal(rowIdx, colIdx))
            V->SetLocal(V->LocalRow(rowIdx), V->LocalCol(colIdx), rightEigs(rowIdx,colIdx));
      // populate S, Sinv
      for(El::Int idx=0; idx < (El::Int) nconv; idx++) {
        if(S->IsLocal(idx, 0))
          S->SetLocal(S->LocalRow(idx), 0, std::sqrt(singValsSq(idx)));
        if(Sinv->IsLocal(idx, 0))
          Sinv->SetLocal(Sinv->LocalRow(idx), 0, 1/std::sqrt(singValsSq(idx)));
      }

      // form U
      El::Gemm(El::NORMAL, El::NORMAL, 1.0, *A, *V, 0.0, *U);
      // TODO: do a QR instead, but does column pivoting so would require postprocessing S,V to stay consistent
      El::DiagonalScale(El::RIGHT, El::NORMAL, *Sinv, *U);

      break;
    }
  }

  self->world.barrier();
}

void TransposeCommand::run(Worker *self) const {
  auto m = self->matrices[origMat]->Height();
  auto n = self->matrices[origMat]->Width();
  DistMatrix * transposeA = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, m, self->grid);
  El::Zero(*transposeA);

  ENSURE(self->matrices.insert(std::make_pair(transposeMat, std::unique_ptr<DistMatrix>(transposeA))).second);

  El::Transpose(*self->matrices[origMat], *transposeA);
  std::cerr << format("%s: finished transpose call\n") % self->world.rank();
  self->world.barrier();
}

void ThinSVDCommand::run(Worker *self) const {
  auto m = self->matrices[mat]->Height();
  auto n = self->matrices[mat]->Width();
  auto k = std::min(m, n);
  DistMatrix * U = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, k, self->grid);
  DistMatrix * singvals = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(k, k, self->grid);
  DistMatrix * V = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, k, self->grid);
  El::Zero(*U);
  El::Zero(*V);
  El::Zero(*singvals);

  ENSURE(self->matrices.insert(std::make_pair(Uhandle, std::unique_ptr<DistMatrix>(U))).second);
  ENSURE(self->matrices.insert(std::make_pair(Shandle, std::unique_ptr<DistMatrix>(singvals))).second);
  ENSURE(self->matrices.insert(std::make_pair(Vhandle, std::unique_ptr<DistMatrix>(V))).second);

  DistMatrix * Acopy = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, n, self->grid); // looking at source code for SVD, seems that DistMatrix Acopy(A) might generate copy rather than just copy metadata and risk clobbering
  El::Copy(*self->matrices[mat], *Acopy);
  El::SVD(*Acopy, *U, *singvals, *V);
  std::cerr << format("%s: singvals is %s by %s\n") % self->world.rank() % singvals->Height() % singvals->Width();
  self->world.barrier();
}

void MatrixMulCommand::run(Worker *self) const {
  auto m = self->matrices[inputA]->Height();
  auto n = self->matrices[inputB]->Width();
  DistMatrix * matrix = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, n, self->grid);
  ENSURE(self->matrices.insert(std::make_pair(handle, std::unique_ptr<DistMatrix>(matrix))).second);
  El::Gemm(El::NORMAL, El::NORMAL, 1.0, *self->matrices[inputA], *self->matrices[inputB], 0.0, *matrix);
  //El::Display(*self->matrices[inputA], "A:");
  //El::Display(*self->matrices[inputB], "B:");
  //El::Display(*matrix, "A*B:");
  self->world.barrier();
}

// TODO: should send back blocks of rows instead of rows? maybe conversion on other side is cheaper?
void  MatrixGetRowsCommand::run(Worker * self) const {
  uint64_t numRowsFromMe = std::count(layout.begin(), layout.end(), self->id);
  auto matrix = self->matrices[handle].get();
  uint64_t numCols = matrix->Width();

  std::vector<uint64_t> localRowIndices; // maps rows in the matrix to rows in the local storage
  std::vector<double> localData(numCols * numRowsFromMe);

  localRowIndices.reserve(numRowsFromMe);
  matrix->ReservePulls(numCols * numRowsFromMe);
  for(uint64_t curRowIdx = 0; localRowIndices.size() < numRowsFromMe; curRowIdx++) {
    if( layout[curRowIdx] == self->id ) {
      localRowIndices.push_back(curRowIdx);
      for(uint64_t col = 0; col < numCols; col++) {
        matrix->QueuePull(curRowIdx, col);
      }
    }
  }
  matrix->ProcessPullQueue(&localData[0]);

  self->sendMatrixRows(handle, matrix->Width(), layout, localRowIndices, localData);
  self->world.barrier();
}

void NewMatrixCommand::run(Worker *self) const {
  auto handle = info.handle;
  self->log->info("Creating new distributed matrix");
  DistMatrix *matrix = new El::DistMatrix<double, El::MD, El::STAR>(info.numRows, info.numCols, self->grid);
  Zero(*matrix);
  ENSURE(self->matrices.insert(std::make_pair(handle, std::unique_ptr<DistMatrix>(matrix))).second);
  self->log->info("Created new distributed matrix");

  std::vector<uint64_t> rowsOnWorker;
  self->log->info("Creating vector of local rows");
  rowsOnWorker.reserve(info.numRows);
  for(El::Int rowIdx = 0; rowIdx < info.numRows; ++rowIdx)
    if (matrix->IsLocalRow(rowIdx))
      rowsOnWorker.push_back(rowIdx);

  for(int workerIdx = 1; workerIdx < self->world.size(); workerIdx++) {
    if( self->world.rank() == workerIdx ) {
      self->world.send(0, 0, rowsOnWorker);
    }
    self->world.barrier();
  }

  self->log->info("Starting to recieve my rows");
  self->receiveMatrixBlocks(handle);
  self->log->info("Received all my matrix rows");
  self->world.barrier();
}


























// From DRIVER

void Driver::handle_getTranspose() {
  MatrixHandle inputMat{input.readInt()};
  log->info("Constructing the transpose of matrix {}", inputMat);

  auto numRows = matrices[inputMat].numCols;
  auto numCols = matrices[inputMat].numRows;
  MatrixHandle transposeHandle = registerMatrix(numRows, numCols);
  TransposeCommand cmd(inputMat, transposeHandle);
  issue(cmd);

  world.barrier(); // wait for command to finish
  output.writeInt(0x1);
  output.writeInt(transposeHandle.id);
  log->info("Wrote handle for transpose");
  output.writeInt(0x1);
  output.flush();
}

// CAVEAT: Assumes tall-and-skinny for now, doesn't allow many options for controlling
// LIMITATIONS: assumes V small enough to fit on one machine (so can use ARPACK instead of PARPACK), but still distributes U,S,V and does distributed computations not needed
void Driver::handle_truncatedSVD() {
  log->info("Starting truncated SVD computation");
  MatrixHandle inputMat{input.readInt()};
  uint32_t k = input.readInt();

  MatrixHandle UHandle{nextMatrixId++};
  MatrixHandle SHandle{nextMatrixId++};
  MatrixHandle VHandle{nextMatrixId++};

  auto m = matrices[inputMat].numRows;
  auto n = matrices[inputMat].numCols;
  TruncatedSVDCommand cmd(inputMat, UHandle, SHandle, VHandle, k);
  issue(cmd);

  ARrcSymStdEig<double> prob(n, k, "LM");
  uint32_t command;
  std::vector<double> zerosVector(n);
  for(uint32_t idx = 0; idx < n; idx++)
    zerosVector[idx] = 0;

  int iterNum = 0;

  while (!prob.ArnoldiBasisFound()) {
    prob.TakeStep();
    ++iterNum;
    if(iterNum % 20 == 0) {
        log->info("Computed {} mv products", iterNum);
    }
    if (prob.GetIdo() == 1 || prob.GetIdo() == -1) {
      command = 1;
      mpi::broadcast(world, command, 0);
      mpi::broadcast(world, prob.GetVector(), n, 0);
      mpi::reduce(world, zerosVector.data(), n, prob.PutVector(), std::plus<double>(), 0);
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
  mpi::broadcast(world, command, 0);
  mpi::broadcast(world, nconv, 0);
  log->info("Broadcasted command and number of converged eigenvectors");
  mpi::broadcast(world, rightVecs.data(), n*nconv, 0);
  log->info("Broadcasted right eigenvectors");
  mpi::broadcast(world, prob.RawEigenvalues(), nconv, 0);
  log->info("Broadcasted eigenvalues");

  MatrixDescriptor Uinfo(UHandle, m, nconv);
  MatrixDescriptor Sinfo(SHandle, nconv, 1);
  MatrixDescriptor Vinfo(VHandle, n, nconv);
  ENSURE(matrices.insert(std::make_pair(UHandle, Uinfo)).second);
  ENSURE(matrices.insert(std::make_pair(SHandle, Sinfo)).second);
  ENSURE(matrices.insert(std::make_pair(VHandle, Vinfo)).second);

  world.barrier();
  log->info("Writing ok status followed by U,S,V handles");
  output.writeInt(0x1);
  output.writeInt(UHandle.id);
  output.writeInt(SHandle.id);
  output.writeInt(VHandle.id);
  output.flush();
}

void Driver::handle_computeThinSVD() {
  MatrixHandle inputMat{input.readInt()};

  // this needs to be done automatically rather than hand-coded. e.g. what if
  // we switch to determining rank by sing-val thresholding instead of doing thin SVD?
  auto m = matrices[inputMat].numRows;
  auto n = matrices[inputMat].numCols;
  auto k = std::min(m,n);
  MatrixHandle Uhandle = registerMatrix(m, k);
  MatrixHandle Shandle = registerMatrix(k, 1);
  MatrixHandle Vhandle = registerMatrix(n, k);
  ThinSVDCommand cmd(inputMat, Uhandle, Shandle, Vhandle);
  issue(cmd);

  output.writeInt(0x1); // statusCode
  output.writeInt(Uhandle.id);
  output.writeInt(Shandle.id);
  output.writeInt(Vhandle.id);
  output.flush();

  // wait for command to finish
  world.barrier();
  log->info("Done with SVD computation");
  output.writeInt(0x1);
  output.flush();
}

void Driver::handle_matrixMul() {
  MatrixHandle matA{input.readInt()};
  MatrixHandle matB{input.readInt()};
  log->info("Multiplying matrices {} and {}", matA, matB);

  auto numRows = matrices[matA].numRows;
  auto numCols = matrices[matB].numCols;
  MatrixHandle destHandle = registerMatrix(numRows, numCols);
  MatrixMulCommand cmd(destHandle, matA, matB);
  issue(cmd);

  // tell spark id of resulting matrix
  output.writeInt(0x1); // statusCode
  output.writeInt(destHandle.id);
  output.flush();

  // wait for it to finish
  world.barrier();
  output.writeInt(0x1);
  output.flush();
}

void Driver::handle_matrixDims() {
  MatrixHandle matrixHandle{input.readInt()};
  auto info = matrices[matrixHandle];
  output.writeInt(0x1);
  output.writeLong(info.numRows);
  output.writeLong(info.numCols);
  output.flush();

}

void Driver::handle_getMatrixRows() {
  MatrixHandle handle{input.readInt()};
  uint64_t layoutLen = input.readLong();
  std::vector<uint32_t> layout;
  layout.reserve(layoutLen);
  for(uint64_t part = 0; part < layoutLen; ++part) {
    layout.push_back(input.readInt());
  }
  log->info("Returning matrix {} to Spark", handle);

  MatrixGetRowsCommand cmd(handle, layout);
  issue(cmd);

  // tell Spark to start asking for rows
  output.writeInt(0x1);
  output.flush();

  // wait for it to finish
  world.barrier();
  output.writeInt(0x1);
  output.flush();
}

void Driver::handle_newMatrix() {
  // read args
  uint64_t numRows = input.readLong();
  uint64_t numCols = input.readLong();

  // assign id and notify workers
  MatrixHandle handle = registerMatrix(numRows, numCols);
  NewMatrixCommand cmd(matrices[handle]);
  log->info("Recieving new matrix {}, with dimensions {}x{}", handle, numRows, numCols);
  issue(cmd);

  output.writeInt(0x1);
  output.writeInt(handle.id);
  output.flush();

  // tell spark which worker expects each row
  std::vector<int> rowWorkerAssignments(numRows, 0);
  std::vector<uint64_t> rowsOnWorker;
  for(int workerIdx = 1; workerIdx < world.size(); workerIdx++) {
    world.recv(workerIdx, 0, rowsOnWorker);
    world.barrier();
    for(auto rowIdx: rowsOnWorker) {
      rowWorkerAssignments[rowIdx] = workerIdx;
    }
  }

  log->info("Sending list of which worker each row should go to");
  output.writeInt(0x1); // statusCode
  for(auto workerIdx: rowWorkerAssignments)
    output.writeInt(workerIdx);
  output.flush();

  log->info("Waiting for spark to finish sending data to the workers");
  world.barrier();
  output.writeInt(0x1);  // statusCode
  output.flush();
  log->info("Entire matrix has been received");
}

