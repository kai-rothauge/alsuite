#include "alML.h"
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <poll.h>
#include "data_stream.h"
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "spdlog/spdlog.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace alML {

class KMeans : public Clustering {
public:
	KMeans() {
		k                   = 2;
		maxIterations       = 20;
		initializationMode  = KMeans.K_MEANS_PARALLEL;
		initializationSteps = 2;
		epsilon             = 1e-4;
		seed                = 10;
	}

	KMeans(int k_, int maxIterations_, string initializationMode_, int initializationSteps_,
			double epsilon_, long seed_) {
		k                   = setK(k_);
		maxIterations       = maxIterations_;
		initializationMode  = initializationMode_;
		initializationSteps = initializationSteps_;
		epsilon             = epsilon_;
		seed                = seed_;
	}

	int getK() {
		return k;
	}

	void setK(int k_) {
		k = (k_ > 1) ? k_ : 2
	}

	int getMaxIterations() {
		return maxIterations;
	}

	void setMaxIterations(int maxIterations_) {
		maxIterations = (maxIterations_ > 1) ? maxIterations_ : 2;
	}

	string getInitializationMode() {
		return initializationMode;
	}

	void setInitializationMode(string initializationMode_) {
		transform(initializationMode_.begin(), initializationMode_.end(), initializationMode_.begin(), ::tolower);
		if strcmp(initializationMode_, "random") || strcmp(initializationMode_, "parallel")
			initializationMode = initializationMode_;
		else
			throw Exception("Unknown initialization for K-Means clustering")
	}

	int getInitializationSteps() {
		return initializationSteps;
	}

	void setInitializationSteps(int initializationSteps_) {
		initializationSteps = (initializationSteps_ > 1) ? initializationSteps_ : 2;
	}

	double getEpsilon() {
		return epsilon;
	}

	void setEpsilon(double epsilon_) {
		epsilon = (epsilon_ > 0.0) ? epsilon_ : 1e-4;
	}

	long getSeed() {
		return seed;
	}

	void setSeed(long seed_) {
		seed = (seed_ > 0) ? seed_ : 1;
	}

	void initialize() {
		strcmp(initializationMode, "random") ? initialize_random() : initialize_parallel();
	}

	void train(int k_, int maxIterations_, string initializationMode_, long seed_) {
		data: RDD[Vector],
		      k: Int,
		      maxIterations: Int,
		      initializationMode: String,
		      seed: Long): KMeansModel = {
		    new KMeans().setK(k)
		      .setMaxIterations(maxIterations)
		      .setInitializationMode(initializationMode)
		      .setSeed(seed)
		      .run(data)
	}

	int findClosest() {

	}

	void run(data
    data: RDD[Vector],
    instr: Option[Instrumentation[NewKMeans]]): KMeansModel = {

	  if (data.getStorageLevel == StorageLevel.NONE) {
		logWarning("The input data is not directly cached, which may hurt performance if its"
		  + " parent RDDs are also uncached.")
	  }

	  // Compute squared norms and cache them.
	  val norms = data.map(Vectors.norm(_, 2.0))
	  norms.persist()
	  val zippedData = data.zip(norms).map { case (v, norm) =>
		new VectorWithNorm(v, norm)
	  }
	  val model = runAlgorithm(zippedData, instr)
	  norms.unpersist()

	  // Warn at the end of the run as well, for increased visibility.
	  if (data.getStorageLevel == StorageLevel.NONE) {
		logWarning("The input data was not directly cached, which may hurt performance if its"
		  + " parent RDDs are also uncached.")
	  }
	  model
	}

	/**
	 * Implementation of K-Means algorithm.
	 */
    private def runAlgorithm(Worker *self) const {
	  auto log = self->log;
	  log->info("Started kmeans");
	  auto origDataMat = self->matrices[origMat].get();
	  auto n = origDataMat->Height();
	  auto d = origDataMat->Width();

	  // relayout matrix if needed so that it is in row-partitioned format
	  // cf http://libelemental.org/pub/slides/ICS13.pdf slide 19 for the cost of redistribution
	  auto distData = origDataMat->DistData();
	  DistMatrix * dataMat = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(n, d, self->grid);
	  if (distData.colDist == El::MD && distData.rowDist == El::STAR) {
	   dataMat = origDataMat;
	  } else {
		auto relayoutStart = std::chrono::system_clock::now();
		El::Copy(*origDataMat, *dataMat); // relayouts data so it is row-wise partitioned
		std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - relayoutStart);
		log->info("Detected matrix is not row-partitioned, so relayouted to row-partitioned; took {} ms ", relayoutDuration.count());
	  }

	  // TODO: store these as local matrices on the driver
	  DistMatrix * centers = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(numCenters, d, self->grid);
	  DistMatrix * assignments = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(n, 1, self->grid);
	  ENSURE(self->matrices.insert(std::make_pair(centersHandle, std::unique_ptr<DistMatrix>(centers))).second);
	  ENSURE(self->matrices.insert(std::make_pair(assignmentsHandle, std::unique_ptr<DistMatrix>(assignments))).second);

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

	  MatrixXd clusterCenters(numCenters, d);
	  MatrixXd oldClusterCenters(numCenters, d);

	  // initialize centers using kMeans||
	  uint32_t scale = 2*numCenters;
	  clusterCenters.setZero();
	  kmeansParallelInit(self, dataMat, localData, scale, initSteps, clusterCenters, seed);

	  // TODO: allow to initialize k-means randomly
	  //MatrixXd clusterCenters = MatrixXd::Random(numCenters, d);

	  /******** START Lloyd's iterations ********/
	  // compute the local cluster assignments
	  std::unique_ptr<uint32_t[]> counts{new uint32_t[numCenters]};
	  std::vector<uint32_t> rowAssignments(localData.rows());
	  VectorXd distanceSq(numCenters);
	  double objVal;

	  updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);

	  MatrixXd centersBuf(numCenters, d);
	  std::unique_ptr<uint32_t[]> countsBuf{new uint32_t[numCenters]};
	  uint32_t numChanged = 0;
	  oldClusterCenters = clusterCenters;

	  while(true) {
		uint32_t nextCommand;
		mpi::broadcast(self->world, nextCommand, 0);

		if (nextCommand == 0xf)  // finished iterating
		  break;
		else if (nextCommand == 2) { // encountered an empty cluster, so randomly pick a point in the dataset as that cluster's centroid
		  uint32_t clusterIdx, rowIdx;
		  mpi::broadcast(self->world, clusterIdx, 0);
		  mpi::broadcast(self->world, rowIdx, 0);
		  if (dataMat->IsLocalRow(rowIdx)) {
			auto localRowIdx = dataMat->LocalRow(rowIdx);
			clusterCenters.row(clusterIdx) = localData.row(localRowIdx);
		  }
		  mpi::broadcast(self->peers, clusterCenters, self->peers.rank());
		  updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
		  self->world.barrier();
		  continue;
		}

		/******** do a regular Lloyd's iteration ********/

		// update the centers
		// TODO: locally compute cluster sums and place in clusterCenters
		oldClusterCenters = clusterCenters;
		clusterCenters.setZero();
		for(uint32_t rowIdx = 0; rowIdx < localData.rows(); ++rowIdx)
		  clusterCenters.row(rowAssignments[rowIdx]) += localData.row(rowIdx);

		mpi::all_reduce(self->peers, clusterCenters.data(), numCenters*d, centersBuf.data(), std::plus<double>());
		std::memcpy(clusterCenters.data(), centersBuf.data(), numCenters*d*sizeof(double));
		mpi::all_reduce(self->peers, counts.get(), numCenters, countsBuf.get(), std::plus<uint32_t>());
		std::memcpy(counts.get(), countsBuf.get(), numCenters*sizeof(uint32_t));

		for(uint32_t rowIdx = 0; rowIdx < numCenters; ++rowIdx)
		  if( counts[rowIdx] > 0)
			clusterCenters.row(rowIdx) /= counts[rowIdx];

		// compute new local assignments
		numChanged = updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
		std::cerr << "computed Updated assingments\n" << std::flush;

		// return the number of changed assignments
		mpi::reduce(self->world, numChanged, std::plus<int>(), 0);
		// return the cluster counts
		mpi::reduce(self->world, counts.get(), numCenters, std::plus<uint32_t>(), 0);
		std::cerr << "returned cluster counts\n" << std::flush;
		if (self->world.rank() == 1) {
		  bool movedQ = (clusterCenters - oldClusterCenters).rowwise().norm().minCoeff() > changeThreshold;
		  self->world.send(0, 0, movedQ);
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
	  for(uint32_t clusterIdx = 0; clusterIdx < numCenters; ++clusterIdx)
		if (centers->IsLocalRow(clusterIdx)) {
		  for(El::Int colIdx = 0; colIdx < d; ++colIdx)
			centers->QueueUpdate(clusterIdx, colIdx, clusterCenters(clusterIdx, colIdx));
		}
	  centers->ProcessQueues();
	  std::chrono::duration<double, std::milli> kMeansWrite_duration(std::chrono::system_clock::now() - startKMeansWrite);
	  std::cerr << self->world.rank() << ": writing the k-means centers and assignments took " << kMeansWrite_duration.count() << "ms\n";

	  mpi::reduce(self->world, objVal, std::plus<double>(), 0);
	  self->world.barrier();
	}


	void initialize_random() {

	}

	void initialize_parallel(DistMatrix const * dataMat, MatrixXd const & localData,
			uint32_t scale, uint32_t initSteps, MatrixXd & clusterCenters, uint64_t seed) {

	  auto d = localData.cols();

	  // if you have the initial cluster seed, send it to everyone
	  uint32_t rowIdx;
	  mpi::broadcast(self->world, rowIdx, 0);
	  MatrixXd initialCenter;

	  if (dataMat->IsLocalRow(rowIdx)) {
		auto localRowIdx = dataMat->LocalRow(rowIdx);
		initialCenter = localData.row(localRowIdx);
		int maybe_root = 1;
		int rootProcess;
		mpi::all_reduce(self->peers, self->peers.rank(), rootProcess, std::plus<int>());
		mpi::broadcast(self->peers, initialCenter, self->peers.rank());
	  }
	  else {
		int maybe_root = 0;
		int rootProcess;
		mpi::all_reduce(self->peers, 0, rootProcess, std::plus<int>());
		mpi::broadcast(self->peers, initialCenter, rootProcess);
	  }

	  //in each step, sample 2*k points on average (totalled across the partitions)
	  // with probability proportional to their squared distance from the current
	  // cluster centers and add the sampled points to the set of cluster centers
	  std::vector<double> distSqToCenters(localData.rows());
	  double Z; // normalization constant
	  std::mt19937 gen(seed + self->world.rank());
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
		mpi::all_reduce(self->peers, mpi::inplace_t<double>(Z), std::plus<double>());

		// 2) sample your points accordingly
		std::vector<MatrixXd> localNewCenters;
		for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
		  bool sampledQ = ( dis(gen) < ((double)scale) * distSqToCenters[pointIdx]/Z ) ? true : false;
		  if (sampledQ) {
			localNewCenters.push_back(localData.row(pointIdx));
		  }
		};

		// 3) have each worker broadcast out their sampled points to all other workers,
		// to update each worker's set of centers
		for(uint32_t root= 0; root < self->peers.size(); ++root) {
		  if (root == self->peers.rank()) {
			mpi::broadcast(self->peers, localNewCenters, root);
			initCenters.insert(initCenters.end(), localNewCenters.begin(), localNewCenters.end());
		  } else {
			std::vector<MatrixXd> remoteNewCenters;
			mpi::broadcast(self->peers, remoteNewCenters, root);
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

	  mpi::all_reduce(self->peers, localClusterSizes.data(), localClusterSizes.size(),
		  clusterSizes.data(), std::plus<uint32_t>());

	  // after centers have been sampled, sync back up with the driver,
	  // and send them there for local clustering
	  self->world.barrier();
	  if(self->world.rank() == 1) {
		self->world.send(0, 0, clusterSizes);
		self->world.send(0, 0, initCenters);
	  }
	  self->world.barrier();

	  clusterCenters.setZero();
	  mpi::broadcast(self->world, clusterCenters.data(), clusterCenters.rows()*d, 0);
	}
};


	uint32_t update_assignments_and_counts(MatrixXd const & dataMat, MatrixXd const & centers,
	    uint32_t * cluster_sizes, std::vector<uint32_t> & row_assignments, double & obj_val) {

		uint32_t num_centers = centers.rows();
		VectorXd distance_squared(num_centers);
		El::Int new_assignment;
		uint32_t num_changed = 0;
		obj_val = 0.0;

		for(uint32_t idx = 0; idx < num_centers; ++idx) cluster_sizes[idx] = 0;

		for(El::Int row_index = 0; row_index < data_mat.rows(); ++row_index) {
			for(uint32_t center_idx = 0; center_idx < num_centers; ++center_idx)
				distance_squared[center_idx] = (data_mat.row(row_index) - centers.row(center_idx)).squaredNorm();
			obj_val += distance_squared.minCoeff(&new_assignment);
			if (row_assignments[rowIdx] != new_assignment) num_changed++;
			row_assignments[rowIdx] = new_assignment;
			clusterSizes[row_assignments[rowIdx]] += 1;
		}

		return num_changed;
	}

} // namespace alML




// FROM WORKER


//// TODO: add seed as argument (make sure different workers do different things)
//void kmeansParallelInit(Worker * self, DistMatrix const * dataMat,
//    MatrixXd const & localData, uint32_t scale, uint32_t initSteps, MatrixXd & clusterCenters, uint64_t seed) {
//
//  auto d = localData.cols();
//
//  // if you have the initial cluster seed, send it to everyone
//  uint32_t rowIdx;
//  mpi::broadcast(self->world, rowIdx, 0);
//  MatrixXd initialCenter;
//
//  if (dataMat->IsLocalRow(rowIdx)) {
//    auto localRowIdx = dataMat->LocalRow(rowIdx);
//    initialCenter = localData.row(localRowIdx);
//    int maybe_root = 1;
//    int rootProcess;
//    mpi::all_reduce(self->peers, self->peers.rank(), rootProcess, std::plus<int>());
//    mpi::broadcast(self->peers, initialCenter, self->peers.rank());
//  }
//  else {
//    int maybe_root = 0;
//    int rootProcess;
//    mpi::all_reduce(self->peers, 0, rootProcess, std::plus<int>());
//    mpi::broadcast(self->peers, initialCenter, rootProcess);
//  }
//
//  //in each step, sample 2*k points on average (totalled across the partitions)
//  // with probability proportional to their squared distance from the current
//  // cluster centers and add the sampled points to the set of cluster centers
//  std::vector<double> distSqToCenters(localData.rows());
//  double Z; // normalization constant
//  std::mt19937 gen(seed + self->world.rank());
//  std::uniform_real_distribution<double> dis(0, 1);
//
//  std::vector<MatrixXd> initCenters;
//  initCenters.push_back(initialCenter);
//
//  for(int steps = 0; steps < initSteps; ++steps) {
//    // 1) compute the distance of your points from the set of centers and all_reduce
//    // to get the normalization for the sampling probability
//    VectorXd distSq(initCenters.size());
//    Z = 0;
//    for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
//      for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
//        distSq[centerIdx] = (localData.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
//      distSqToCenters[pointIdx] = distSq.minCoeff();
//      Z += distSqToCenters[pointIdx];
//    }
//    mpi::all_reduce(self->peers, mpi::inplace_t<double>(Z), std::plus<double>());
//
//    // 2) sample your points accordingly
//    std::vector<MatrixXd> localNewCenters;
//    for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
//      bool sampledQ = ( dis(gen) < ((double)scale) * distSqToCenters[pointIdx]/Z ) ? true : false;
//      if (sampledQ) {
//        localNewCenters.push_back(localData.row(pointIdx));
//      }
//    };
//
//    // 3) have each worker broadcast out their sampled points to all other workers,
//    // to update each worker's set of centers
//    for(uint32_t root= 0; root < self->peers.size(); ++root) {
//      if (root == self->peers.rank()) {
//        mpi::broadcast(self->peers, localNewCenters, root);
//        initCenters.insert(initCenters.end(), localNewCenters.begin(), localNewCenters.end());
//      } else {
//        std::vector<MatrixXd> remoteNewCenters;
//        mpi::broadcast(self->peers, remoteNewCenters, root);
//        initCenters.insert(initCenters.end(), remoteNewCenters.begin(), remoteNewCenters.end());
//      }
//    }
//  } // end for
//
//  // figure out the number of points closest to each cluster center
//  std::vector<uint32_t> clusterSizes(initCenters.size(), 0);
//  std::vector<uint32_t> localClusterSizes(initCenters.size(), 0);
//  VectorXd distSq(initCenters.size());
//  for(uint32_t pointIdx = 0; pointIdx < localData.rows(); ++pointIdx) {
//    for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
//      distSq[centerIdx] = (localData.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
//    uint32_t clusterIdx;
//    distSq.minCoeff(&clusterIdx);
//    localClusterSizes[clusterIdx] += 1;
//  }
//
//  mpi::all_reduce(self->peers, localClusterSizes.data(), localClusterSizes.size(),
//      clusterSizes.data(), std::plus<uint32_t>());
//
//  // after centers have been sampled, sync back up with the driver,
//  // and send them there for local clustering
//  self->world.barrier();
//  if(self->world.rank() == 1) {
//    self->world.send(0, 0, clusterSizes);
//    self->world.send(0, 0, initCenters);
//  }
//  self->world.barrier();
//
//  clusterCenters.setZero();
//  mpi::broadcast(self->world, clusterCenters.data(), clusterCenters.rows()*d, 0);
//}
//
//// TODO: add seed as argument (make sure different workers do different things)
//void KMeansCommand::run(Worker *self) const {
//  auto log = self->log;
//  log->info("Started kmeans");
//  auto origDataMat = self->matrices[origMat].get();
//  auto n = origDataMat->Height();
//  auto d = origDataMat->Width();
//
//  // relayout matrix if needed so that it is in row-partitioned format
//  // cf http://libelemental.org/pub/slides/ICS13.pdf slide 19 for the cost of redistribution
//  auto distData = origDataMat->DistData();
//  DistMatrix * dataMat = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(n, d, self->grid);
//  if (distData.colDist == El::MD && distData.rowDist == El::STAR) {
//   dataMat = origDataMat;
//  } else {
//    auto relayoutStart = std::chrono::system_clock::now();
//    El::Copy(*origDataMat, *dataMat); // relayouts data so it is row-wise partitioned
//    std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - relayoutStart);
//    log->info("Detected matrix is not row-partitioned, so relayouted to row-partitioned; took {} ms ", relayoutDuration.count());
//  }
//
//  // TODO: store these as local matrices on the driver
//  DistMatrix * centers = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(numCenters, d, self->grid);
//  DistMatrix * assignments = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(n, 1, self->grid);
//  ENSURE(self->matrices.insert(std::make_pair(centersHandle, std::unique_ptr<DistMatrix>(centers))).second);
//  ENSURE(self->matrices.insert(std::make_pair(assignmentsHandle, std::unique_ptr<DistMatrix>(assignments))).second);
//
//  MatrixXd localData(dataMat->LocalHeight(), d);
//
//  // compute the map from local row indices to the row indices in the global matrix
//  // and populate the local data matrix
//
//  std::vector<El::Int> rowMap(localData.rows());
//  for(El::Int rowIdx = 0; rowIdx < n; ++rowIdx)
//    if (dataMat->IsLocalRow(rowIdx)) {
//      auto localRowIdx = dataMat->LocalRow(rowIdx);
//      rowMap[localRowIdx] = rowIdx;
//      for(El::Int colIdx = 0; colIdx < d; ++colIdx)
//        localData(localRowIdx, colIdx) = dataMat->GetLocal(localRowIdx, colIdx);
//    }
//
//  MatrixXd clusterCenters(numCenters, d);
//  MatrixXd oldClusterCenters(numCenters, d);
//
//  // initialize centers using kMeans||
//  uint32_t scale = 2*numCenters;
//  clusterCenters.setZero();
//  kmeansParallelInit(self, dataMat, localData, scale, initSteps, clusterCenters, seed);
//
//  // TODO: allow to initialize k-means randomly
//  //MatrixXd clusterCenters = MatrixXd::Random(numCenters, d);
//
//  /******** START Lloyd's iterations ********/
//  // compute the local cluster assignments
//  std::unique_ptr<uint32_t[]> counts{new uint32_t[numCenters]};
//  std::vector<uint32_t> rowAssignments(localData.rows());
//  VectorXd distanceSq(numCenters);
//  double objVal;
//
//  updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
//
//  MatrixXd centersBuf(numCenters, d);
//  std::unique_ptr<uint32_t[]> countsBuf{new uint32_t[numCenters]};
//  uint32_t numChanged = 0;
//  oldClusterCenters = clusterCenters;
//
//  while(true) {
//    uint32_t nextCommand;
//    mpi::broadcast(self->world, nextCommand, 0);
//
//    if (nextCommand == 0xf)  // finished iterating
//      break;
//    else if (nextCommand == 2) { // encountered an empty cluster, so randomly pick a point in the dataset as that cluster's centroid
//      uint32_t clusterIdx, rowIdx;
//      mpi::broadcast(self->world, clusterIdx, 0);
//      mpi::broadcast(self->world, rowIdx, 0);
//      if (dataMat->IsLocalRow(rowIdx)) {
//        auto localRowIdx = dataMat->LocalRow(rowIdx);
//        clusterCenters.row(clusterIdx) = localData.row(localRowIdx);
//      }
//      mpi::broadcast(self->peers, clusterCenters, self->peers.rank());
//      updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
//      self->world.barrier();
//      continue;
//    }
//
//    /******** do a regular Lloyd's iteration ********/
//
//    // update the centers
//    // TODO: locally compute cluster sums and place in clusterCenters
//    oldClusterCenters = clusterCenters;
//    clusterCenters.setZero();
//    for(uint32_t rowIdx = 0; rowIdx < localData.rows(); ++rowIdx)
//      clusterCenters.row(rowAssignments[rowIdx]) += localData.row(rowIdx);
//
//    mpi::all_reduce(self->peers, clusterCenters.data(), numCenters*d, centersBuf.data(), std::plus<double>());
//    std::memcpy(clusterCenters.data(), centersBuf.data(), numCenters*d*sizeof(double));
//    mpi::all_reduce(self->peers, counts.get(), numCenters, countsBuf.get(), std::plus<uint32_t>());
//    std::memcpy(counts.get(), countsBuf.get(), numCenters*sizeof(uint32_t));
//
//    for(uint32_t rowIdx = 0; rowIdx < numCenters; ++rowIdx)
//      if( counts[rowIdx] > 0)
//        clusterCenters.row(rowIdx) /= counts[rowIdx];
//
//    // compute new local assignments
//    numChanged = updateAssignmentsAndCounts(localData, clusterCenters, counts.get(), rowAssignments, objVal);
//    std::cerr << "computed Updated assingments\n" << std::flush;
//
//    // return the number of changed assignments
//    mpi::reduce(self->world, numChanged, std::plus<int>(), 0);
//    // return the cluster counts
//    mpi::reduce(self->world, counts.get(), numCenters, std::plus<uint32_t>(), 0);
//    std::cerr << "returned cluster counts\n" << std::flush;
//    if (self->world.rank() == 1) {
//      bool movedQ = (clusterCenters - oldClusterCenters).rowwise().norm().minCoeff() > changeThreshold;
//      self->world.send(0, 0, movedQ);
//    }
//  }
//
//  // write the final k-means centers and assignments
//  auto startKMeansWrite = std::chrono::system_clock::now();
//  El::Zero(*assignments);
//  assignments->Reserve(localData.rows());
//  for(El::Int rowIdx = 0; rowIdx < localData.rows(); ++rowIdx)
//    assignments->QueueUpdate(rowMap[rowIdx], 0, rowAssignments[rowIdx]);
//  assignments->ProcessQueues();
//
//  El::Zero(*centers);
//  centers->Reserve(centers->LocalHeight()*d);
//  for(uint32_t clusterIdx = 0; clusterIdx < numCenters; ++clusterIdx)
//    if (centers->IsLocalRow(clusterIdx)) {
//      for(El::Int colIdx = 0; colIdx < d; ++colIdx)
//        centers->QueueUpdate(clusterIdx, colIdx, clusterCenters(clusterIdx, colIdx));
//    }
//  centers->ProcessQueues();
//  std::chrono::duration<double, std::milli> kMeansWrite_duration(std::chrono::system_clock::now() - startKMeansWrite);
//  std::cerr << self->world.rank() << ": writing the k-means centers and assignments took " << kMeansWrite_duration.count() << "ms\n";
//
//  mpi::reduce(self->world, objVal, std::plus<double>(), 0);
//  self->world.barrier();
//}


















// FROM DRIVER


// TODO: the cluster centers should be stored locally on driver and reduced/broadcasted. the current
// way of updating kmeans centers is ridiculous
// TODO: currently only implements kmeans||
void Driver::handle_kmeansClustering() {
  MatrixHandle inputMat{input.readInt()};
  uint32_t numCenters = input.readInt();
  uint32_t maxnumIters = input.readInt(); // how many iteration of Lloyd's algorithm to use
  uint32_t initSteps = input.readInt(); // number of initialization steps to use in kmeans||
  double changeThreshold = input.readDouble(); // if all the centers change by Euclidean distance less than changeThreshold, then we stop the iterations
  uint32_t method = input.readInt(); // which initialization method to use to choose initial cluster center guesses
  uint64_t seed = input.readLong(); // randomness seed used in driver and workers

  log->info("Starting K-means on matrix {}", inputMat);
  log->info("numCenters = {}, maxnumIters = {}, initSteps = {}, changeThreshold = {}, method = {}, seed = {}",
      numCenters, maxnumIters, initSteps, changeThreshold, method, seed);

  if (changeThreshold < 0.0 || changeThreshold > 1.0) {
    log->error("Unreasonable change threshold in k-means: {}", changeThreshold);
    abort();
  }
  if (method != 1) {
    log->warn("Sorry, only k-means|| initialization has been implemented, so ignoring your choice of method {}", method);
  }

  auto n = matrices[inputMat].numRows;
  auto d = matrices[inputMat].numCols;
  MatrixHandle centersHandle = this->registerMatrix(numCenters, d);
  MatrixHandle assignmentsHandle = this->registerMatrix(n, 1);
  KMeansCommand cmd(inputMat, numCenters, method, initSteps, changeThreshold, seed, centersHandle, assignmentsHandle);
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
  MatrixXd clusterCenters(numCenters, d);

  kmeansPP(gen(), initClusterCenters, weights, clusterCenters, 30); // same number of maxIters as spark kmeans

  log->info("Ran local k-means on the driver to determine starting cluster centers");
  log->debug("{}", clusterCenters);

  mpi::broadcast(world, clusterCenters.data(), numCenters*d, 0);
  /******** END of kMeans|| initialization ********/

  /******** START of Lloyd's algorithm iterations ********/
  double percentAssignmentsChanged = 1.0;
  bool centersMovedQ = true;
  uint32_t numChanged = 0;
  uint32_t numIters = 0;
  std::vector<uint32_t> parClusterSizes(numCenters);
  std::vector<uint32_t> zerosVector(numCenters);

  for(uint32_t clusterIdx = 0; clusterIdx < numCenters; clusterIdx++)
    zerosVector[clusterIdx] = 0;

  uint32_t command = 1; // do another iteration
  while (centersMovedQ && numIters++ < maxnumIters)  {
    log->info("Starting iteration {} of Lloyd's algorithm, {} percentage changed in last iter",
        numIters, percentAssignmentsChanged*100);
    numChanged = 0;
    for(uint32_t clusterIdx = 0; clusterIdx < numCenters; clusterIdx++)
      parClusterSizes[clusterIdx] = 0;
    command = 1; // do a basic iteration
    mpi::broadcast(world, command, 0);
    mpi::reduce(world, (uint32_t) 0, numChanged, std::plus<int>(), 0);
    mpi::reduce(world, zerosVector.data(), numCenters, parClusterSizes.data(), std::plus<uint32_t>(), 0);
    world.recv(1, mpi::any_tag, centersMovedQ);
    percentAssignmentsChanged = ((double) numChanged)/n;

    for(uint32_t clusterIdx = 0; clusterIdx < numCenters; clusterIdx++) {
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


