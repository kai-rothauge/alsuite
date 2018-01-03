#include "clustering.hpp"

namespace allib {

//KMeans::KMeans() : num_centers(2), max_iterations(20), epsilon(1e-4), init_mode("kmeans||"), init_steps(2), seed(10), data_matrix(nullptr) { }

//KMeans::KMeans(uint32_t _num_centers, uint32_t _max_iterations, double _epsilon, std::string _init_mode,
//			uint32_t _init_steps, uint64_t _seed) :
//		num_centers(_num_centers), max_iterations(_max_iterations), epsilon(_epsilon), init_mode(_init_mode),
//		init_steps(_init_steps), seed(_seed), data_matrix(nullptr)
//{
//	if (epsilon < 0.0 || epsilon > 1.0) {
//		log->error("Unreasonable change threshold in k-means: {}", epsilon);
//		epsilon = 1e-4;
//		log->error("Change threshold changed to default value {}", 1e-4);
//	}
//
//	if strcmp(init_mode, "kmeans||") {
//		init_mode = "kmeans||";
//		log->warn("Sorry, only k-means|| initialization has been implemented, so ignoring your choice of method {}", init_mode);
//	}
//}

void KMeans::set_parameters(uint32_t _num_centers, uint32_t _max_iterations, double _epsilon, std::string _init_mode,
		uint32_t _init_steps, uint64_t _seed)
{
	num_centers = _num_centers;
	max_iterations = _max_iterations;
	epsilon = _epsilon;
	init_mode = _init_mode;
	init_steps = _init_steps;
	seed = _seed;
	data = nullptr;

	if (epsilon < 0.0 || epsilon > 1.0) {
		log->error("Unreasonable change threshold in k-means: {}", epsilon);
		epsilon = 1e-4;
		log->error("Change threshold changed to default value {}", 1e-4);
	}

	if (strcmp(init_mode.c_str(), "kmeans||")) {
		init_mode = "kmeans||";
		log->warn("Sorry, only k-means|| initialization has been implemented, so ignoring your choice of method {}", init_mode);
	}
}

uint32_t KMeans::get_num_centers() {
	return num_centers;
}

void KMeans::set_num_centers(uint32_t _num_centers) {
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

void KMeans::set_init_steps(uint32_t _init_steps) {
	init_steps = _init_steps;
}

double KMeans::get_epsilon() {
	return epsilon;
}

void KMeans::set_epsilon(double _epsilon) {
	epsilon = _epsilon;
}

uint64_t KMeans::get_seed() {
	return seed;
}

void KMeans::set_seed(uint64_t _seed) {
	seed = _seed;
}

int KMeans::initialize(DistMatrix const * data, MatrixXd const & local_data, uint32_t scale, MatrixXd & cluster_centers) {
	strcmp(init_mode.c_str(), "random") ? initialize_random() : initialize_parallel(local_data, scale, cluster_centers);
}

int KMeans::run(Parameters & output) {
	log->info("Settings:");
	train(output);
}

uint32_t KMeans::update_assignments_and_counts(MatrixXd const & data, MatrixXd const & centers,
    uint32_t * cluster_sizes, std::vector<uint32_t> & row_assignments, double & objVal) {

	uint32_t num_centers = centers.rows();
	VectorXd distanceSq(num_centers);
	El::Int newAssignment;
	uint32_t numChanged = 0;
	objVal = 0.0;

	for(uint32_t idx = 0; idx < num_centers; ++idx)
		cluster_sizes[idx] = 0;

	for(El::Int row_index = 0; row_index < data.rows(); ++row_index) {
		for(uint32_t centerIdx = 0; centerIdx < num_centers; ++centerIdx)
			distanceSq[centerIdx] = (data.row(row_index) - centers.row(centerIdx)).squaredNorm();
		objVal += distanceSq.minCoeff(&newAssignment);
		if (row_assignments[row_index] != newAssignment)
			numChanged++;
		row_assignments[row_index] = newAssignment;
		cluster_sizes[row_assignments[row_index]] += 1;
	}

	return numChanged;
}

void KMeans::set_data_matrix(DistMatrix * _data) {
	data = _data;
}

int KMeans::initialize_random() {

	return 0;
}

// TODO: add seed as argument (make sure different workers do different things)
int KMeans::initialize_parallel(MatrixXd const & local_data, uint32_t scale, MatrixXd & cluster_centers) {

	auto d = local_data.cols();

	// if you have the initial cluster seed, send it to everyone
	uint32_t row_index;
	boost::mpi::broadcast(world, row_index, 0);
	MatrixXd initialCenter;

	if (data->IsLocalRow(row_index)) {
		auto localrow_index = data->LocalRow(row_index);
		initialCenter = local_data.row(localrow_index);
		int maybe_root = 1;
		int rootProcess;
		boost::mpi::all_reduce(peers, peers.rank(), rootProcess, std::plus<int>());
		boost::mpi::broadcast(peers, initialCenter, peers.rank());
	}
	else {
		int maybe_root = 0;
		int rootProcess;
		boost::mpi::all_reduce(peers, 0, rootProcess, std::plus<int>());
		boost::mpi::broadcast(peers, initialCenter, rootProcess);
	}

	//in each step, sample 2*k points on average (totalled across the partitions)
	// with probability proportional to their squared distance from the current
	// cluster centers and add the sampled points to the set of cluster centers
	std::vector<double> distSqToCenters(local_data.rows());
	double Z; // normalization constant
	std::mt19937 gen(seed + world.rank());
	std::uniform_real_distribution<double> dis(0, 1);

	std::vector<MatrixXd> initCenters;
	initCenters.push_back(initialCenter);

	for(int steps = 0; steps < init_steps; ++steps) {
		// 1) compute the distance of your points from the set of centers and all_reduce
		// to get the normalization for the sampling probability
		VectorXd distSq(initCenters.size());
		Z = 0;
		for(uint32_t pointIdx = 0; pointIdx < local_data.rows(); ++pointIdx) {
			for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
				distSq[centerIdx] = (local_data.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
			distSqToCenters[pointIdx] = distSq.minCoeff();
			Z += distSqToCenters[pointIdx];
		}
		boost::mpi::all_reduce(peers, boost::mpi::inplace_t<double>(Z), std::plus<double>());

		// 2) sample your points accordingly
		std::vector<MatrixXd> localNewCenters;
		for(uint32_t pointIdx = 0; pointIdx < local_data.rows(); ++pointIdx) {
			bool sampledQ = ( dis(gen) < ((double)scale) * distSqToCenters[pointIdx]/Z ) ? true : false;
			if (sampledQ) {
				localNewCenters.push_back(local_data.row(pointIdx));
			}
		}

		// 3) have each worker broadcast out their sampled points to all other workers,
		// to update each worker's set of centers
		for(uint32_t root= 0; root < peers.size(); ++root) {
			if (root == peers.rank()) {
				boost::mpi::broadcast(peers, localNewCenters, root);
				initCenters.insert(initCenters.end(), localNewCenters.begin(), localNewCenters.end());
			} else {
				std::vector<MatrixXd> remoteNewCenters;
				boost::mpi::broadcast(peers, remoteNewCenters, root);
				initCenters.insert(initCenters.end(), remoteNewCenters.begin(), remoteNewCenters.end());
			}
		}
	} // end for

	// figure out the number of points closest to each cluster center
	std::vector<uint32_t> cluster_sizes(initCenters.size(), 0);
	std::vector<uint32_t> localcluster_sizes(initCenters.size(), 0);
	VectorXd distSq(initCenters.size());
	for(uint32_t pointIdx = 0; pointIdx < local_data.rows(); ++pointIdx) {
		for(uint32_t centerIdx = 0; centerIdx < initCenters.size(); ++centerIdx)
			distSq[centerIdx] = (local_data.row(pointIdx) - initCenters[centerIdx]).squaredNorm();
		uint32_t cluster_index;
		distSq.minCoeff(&cluster_index);
		localcluster_sizes[cluster_index] += 1;
	}

	boost::mpi::all_reduce(peers, localcluster_sizes.data(), localcluster_sizes.size(),
	cluster_sizes.data(), std::plus<uint32_t>());

	// after centers have been sampled, sync back up with the driver,
	// and send them there for local clustering
	world.barrier();
	if (world.rank() == 1) {
		world.send(0, 0, cluster_sizes);
		world.send(0, 0, initCenters);
	}
	world.barrier();

	cluster_centers.setZero();
	boost::mpi::broadcast(world, cluster_centers.data(), cluster_centers.rows()*d, 0);

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
	std::vector<double> cluster_sizes(k, 0.0);
	std::vector<uint32_t> clusterAssignments(n, 0);
	MatrixXd clusterPointSums(k, d);
	VectorXd distanceSqToCenters(k);
	uint32_t newClusterAssignment;
	double sqDist;

	uint32_t iter = 0;
	for(; iter < max_iterations; iter++) {
		movedQ = false;
		clusterPointSums.setZero();
		std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);

		// assign each point to nearest cluster and count number of points in each cluster
		for(auto pointIdx : pointIndices) {
			for(auto centerIdx : centerIndices)
				distanceSqToCenters(centerIdx) = (points[pointIdx] - fitCenters.row(centerIdx)).squaredNorm();
			sqDist = distanceSqToCenters.minCoeff(&newClusterAssignment);
			if (newClusterAssignment != clusterAssignments[pointIdx])
				movedQ = true;
			clusterAssignments[pointIdx] = newClusterAssignment;
			clusterPointSums.row(newClusterAssignment) += weights[pointIdx]*points[pointIdx];
			cluster_sizes[newClusterAssignment] += weights[pointIdx];
		}

		// stop iterations if cluster assignments have not changed
		if(!movedQ) break;

		// update cluster centers
		for(auto centerIdx : centerIndices) {
			if ( cluster_sizes[centerIdx] > 0 ) {
				fitCenters.row(centerIdx) = clusterPointSums.row(centerIdx) / cluster_sizes[centerIdx];
			} else {
				uint32_t randPtIdx = (uint32_t) std::round(unifReal(randGen)*n);
				fitCenters.row(centerIdx) = points[randPtIdx];
			}
		}
	}

	// seems necessary to force eigen to return the centers as an actual usable matrix
	for(uint32_t row_index = 0; row_index < k; row_index++)
		for(uint32_t colidx = 0; colidx < k; colidx++)
			fitCenters(row_index, colidx) = fitCenters(row_index, colidx) + 0.0;

	return 0;
}

int KMeans::train(Parameters & output) {

	uint32_t command = 0;

	if (isDriver) {

		int m, n;

		world.recv(1, 0, m);
		world.recv(1, 0, n);

		world.barrier();

		log->info("Starting K-Means on {}x{} matrix", m, n);
		log->info("Settings:");
		log->info("    num_centers = {}", num_centers);
		log->info("    max_iterations = {}", max_iterations);
		log->info("    epsilon = {}", epsilon);
		log->info("    init_mode = {}", init_mode);
		log->info("    init_steps = {}", init_steps);
		log->info("    seed = {}", seed);

//		MatrixHandle centersHandle = this->registerMatrix(num_centers, d);
//		MatrixHandle assignmentsHandle = this->registerMatrix(n, 1);

		uint32_t command = 0;
		boost::mpi::broadcast(world, command, 0);		// Tell workers to start

		/******** START of kmeans|| initialization ********/
		std::mt19937 gen(seed);
		std::uniform_int_distribution<unsigned long> dis(0, m-1);
		uint32_t row_index = dis(gen);
		std::vector<double> initialCenter(n);

		boost::mpi::broadcast(world, row_index, 0); // tell the workers which row to use as initialization in kmeans||
		world.barrier(); // wait for workers to return oversampled cluster centers and sizes

		std::vector<uint32_t> cluster_sizes;
		std::vector<MatrixXd> init_cluster_centers;
		world.recv(1, boost::mpi::any_tag, cluster_sizes);
		world.recv(1, boost::mpi::any_tag, init_cluster_centers);
		world.barrier();

		log->info("Retrieved the k-means|| oversized set of potential cluster centers");

		// use kmeans++ locally to find the initial cluster centers
		std::vector<double> weights;
		weights.reserve(cluster_sizes.size());
		std::for_each(cluster_sizes.begin(), cluster_sizes.end(), [&weights](const uint32_t & cnt){ weights.push_back((double) cnt); });
		MatrixXd cluster_centers(num_centers, n);

		kmeansPP(init_cluster_centers, weights, cluster_centers); // same number of maxIters as spark kmeans

		log->info("Ran local k-means on the driver to determine starting cluster centers");

		boost::mpi::broadcast(world, cluster_centers.data(), num_centers*n, 0);
		/******** END of kMeans|| initialization ********/

		/******** START of Lloyd's algorithm iterations ********/
		double percentAssignmentsChanged = 1.0;
		bool centersMovedQ = true;
		uint32_t numChanged = 0;
		uint32_t numIters = 0;
		std::vector<uint32_t> parcluster_sizes(num_centers);
		std::vector<uint32_t> zerosVector(num_centers);

		for (uint32_t cluster_index = 0; cluster_index < num_centers; cluster_index++)
			zerosVector[cluster_index] = 0;

		command = 1; // do another iteration
		while (centersMovedQ && numIters++ < max_iterations)  {
			log->info("Starting iteration {} of Lloyd's algorithm, {} percentage changed in last iteration",
			numIters, percentAssignmentsChanged*100);
			numChanged = 0;
			for(uint32_t cluster_index = 0; cluster_index < num_centers; cluster_index++)
				parcluster_sizes[cluster_index] = 0;
			command = 1; // do a basic iteration
			boost::mpi::broadcast(world, command, 0);
			boost::mpi::reduce(world, (uint32_t) 0, numChanged, std::plus<int>(), 0);
			boost::mpi::reduce(world, zerosVector.data(), num_centers, parcluster_sizes.data(), std::plus<uint32_t>(), 0);
			world.recv(1, boost::mpi::any_tag, centersMovedQ);
			percentAssignmentsChanged = ((double) numChanged)/m;

			for(uint32_t cluster_index = 0; cluster_index < num_centers; cluster_index++) {
				if (parcluster_sizes[cluster_index] == 0) {
					// this is an empty cluster, so randomly pick a point in the dataset
					// as that cluster's centroid
					centersMovedQ = true;
					command = 2; // reinitialize this cluster center
					uint32_t row_index = dis(gen);
					boost::mpi::broadcast(world, command, 0);
					boost::mpi::broadcast(world, cluster_index, 0);
					boost::mpi::broadcast(world, row_index, 0);
					world.barrier();
				}
			}
		}
		command = 0xf; // terminate and finalize the k-means centers and assignments as distributed matrices
		boost::mpi::broadcast(world, command, 0);
		double objVal = 0.0;
		boost::mpi::reduce(world, 0.0, objVal, std::plus<double>(), 0);
		world.barrier();

		/******** END of Lloyd's iterations ********/

		log->info("Finished Lloyd's algorithm: took {} iterations, final objective value {}", numIters, objVal);
		output.add_int("num_iterations", numIters);
		output.add_double("objective_value", objVal);
	}
	else {
		int m = data->Height();
		int n = data->Width();

		if (world.rank() == 1) {
			world.send(0, 0, m);
			world.send(0, 0, n);
		}

		world.barrier();

		boost::mpi::broadcast(world, command, 0);

		if (command == 0) {
			log->info("Started K-Means");

			// relayout matrix if needed so that it is in row-partitioned format
			// cf http://libelemental.org/pub/slides/ICS13.pdf slide 19 for the cost of redistribution
			auto distData = data->DistData();
			DistMatrix * new_data = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(m, n, *grid);
			if (distData.colDist == El::MD && distData.rowDist == El::STAR) {
				new_data = data;
			} else {
				auto relayoutStart = std::chrono::system_clock::now();
				El::Copy(*data, *new_data); // relayouts data so it is row-wise partitioned
				DistMatrix * temp = data;
				data = new_data;
				delete(temp);
				std::chrono::duration<double, std::milli> relayoutDuration(std::chrono::system_clock::now() - relayoutStart);
				log->info("Changed layout of data matrix to be row-partitioned; took {} ms ", relayoutDuration.count());
			}

			// TODO: store these as local matrices on the driver
			DistMatrix * centers = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(num_centers, n, *grid);
			DistMatrix * assignments = new El::DistMatrix<double, El::MD, El::STAR, El::ELEMENT>(m, 1, *grid);
//			ENSURE(matrices.insert(std::make_pair(centersHandle, std::unique_ptr<DistMatrix>(centers))).second);
//			ENSURE(matrices.insert(std::make_pair(assignmentsHandle, std::unique_ptr<DistMatrix>(assignments))).second);

			MatrixXd local_data(data->LocalHeight(), n);

			// compute the map from local row indices to the row indices in the global matrix
			// and populate the local data matrix

			std::vector<El::Int> rowMap(local_data.rows());
			for(El::Int row_index = 0; row_index < m; ++row_index)
				if (data->IsLocalRow(row_index)) {
					auto localrow_index = data->LocalRow(row_index);
					rowMap[localrow_index] = row_index;
					for(El::Int colIdx = 0; colIdx < n; ++colIdx)
						local_data(localrow_index, colIdx) = data->GetLocal(localrow_index, colIdx);
				}

			MatrixXd cluster_centers(num_centers, n);
			MatrixXd old_cluster_centers(num_centers, n);

			// initialize centers using kMeans||
			uint32_t scale = 2*num_centers;
			cluster_centers.setZero();
			initialize_parallel(local_data, scale, cluster_centers);

			// TODO: allow to initialize k-means randomly
			//MatrixXd cluster_centers = MatrixXd::Random(num_centers, d);

			/******** START Lloyd's iterations ********/
			// compute the local cluster assignments
			std::unique_ptr<uint32_t[]> counts{new uint32_t[num_centers]};
			std::vector<uint32_t> row_assignments(local_data.rows());
			VectorXd distanceSq(num_centers);
			double objVal;

			update_assignments_and_counts(local_data, cluster_centers, counts.get(), row_assignments, objVal);

			MatrixXd centersBuf(num_centers, n);
			std::unique_ptr<uint32_t[]> countsBuf{new uint32_t[num_centers]};
			uint32_t numChanged = 0;
			old_cluster_centers = cluster_centers;

			while (true) {
				uint32_t nextCommand;
				boost::mpi::broadcast(world, nextCommand, 0);

				if (nextCommand == 0xf)  // finished iterating
					break;
				else if (nextCommand == 2) { // encountered an empty cluster, so randomly pick a point in the dataset as that cluster's centroid
					uint32_t cluster_index, row_index;
					boost::mpi::broadcast(world, cluster_index, 0);
					boost::mpi::broadcast(world, row_index, 0);
					if (data->IsLocalRow(row_index)) {
						auto localrow_index = data->LocalRow(row_index);
						cluster_centers.row(cluster_index) = local_data.row(localrow_index);
					}
					boost::mpi::broadcast(peers, cluster_centers, peers.rank());
					update_assignments_and_counts(local_data, cluster_centers, counts.get(), row_assignments, objVal);
					world.barrier();
					continue;
				}

				/******** do a regular Lloyd's iteration ********/

				// update the centers
				// TODO: locally compute cluster sums and place in cluster_centers
				old_cluster_centers = cluster_centers;
				cluster_centers.setZero();
				for(uint32_t row_index = 0; row_index < local_data.rows(); ++row_index)
					cluster_centers.row(row_assignments[row_index]) += local_data.row(row_index);

				boost::mpi::all_reduce(peers, cluster_centers.data(), num_centers*n, centersBuf.data(), std::plus<double>());
				std::memcpy(cluster_centers.data(), centersBuf.data(), num_centers*n*sizeof(double));
				boost::mpi::all_reduce(peers, counts.get(), num_centers, countsBuf.get(), std::plus<uint32_t>());
				std::memcpy(counts.get(), countsBuf.get(), num_centers*sizeof(uint32_t));

				for(uint32_t row_index = 0; row_index < num_centers; ++row_index)
					if( counts[row_index] > 0)
						cluster_centers.row(row_index) /= counts[row_index];

				// compute new local assignments
				numChanged = update_assignments_and_counts(local_data, cluster_centers, counts.get(), row_assignments, objVal);
				log->info("Updated assignments");

				// return the number of changed assignments
				boost::mpi::reduce(world, numChanged, std::plus<int>(), 0);
				// return the cluster counts
				boost::mpi::reduce(world, counts.get(), num_centers, std::plus<uint32_t>(), 0);
				log->info("Returned cluster counts");
				if (world.rank() == 1) {
					bool movedQ = (cluster_centers - old_cluster_centers).rowwise().norm().minCoeff() > epsilon;
					world.send(0, 0, movedQ);
				}
			}

			// write the final k-means centers and assignments
			auto startKMeansWrite = std::chrono::system_clock::now();
			El::Zero(*assignments);
			assignments->Reserve(local_data.rows());
			for(El::Int row_index = 0; row_index < local_data.rows(); ++row_index)
				assignments->QueueUpdate(rowMap[row_index], 0, row_assignments[row_index]);
			assignments->ProcessQueues();

			El::Zero(*centers);
			centers->Reserve(centers->LocalHeight()*n);
			for(uint32_t cluster_index = 0; cluster_index < num_centers; ++cluster_index)
				if (centers->IsLocalRow(cluster_index)) {
					for(El::Int colIdx = 0; colIdx < n; ++colIdx)
						centers->QueueUpdate(cluster_index, colIdx, cluster_centers(cluster_index, colIdx));
				}
			centers->ProcessQueues();
			std::chrono::duration<double, std::milli> kMeansWrite_duration(std::chrono::system_clock::now() - startKMeansWrite);
			log->info("Writing the k-means centers and assignments took {}", kMeansWrite_duration.count());

			boost::mpi::reduce(world, objVal, std::plus<double>(), 0);
			world.barrier();

			output.add_distmatrix("assignments", assignments);
			output.add_distmatrix("centers", centers);
		}
	}


	return 0;
}

} // namespace allib
