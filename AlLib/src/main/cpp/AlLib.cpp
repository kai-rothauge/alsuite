#include "AlLib.hpp"

using alchemist::Parameters;

namespace allib {

int AlLib::load() {

	bool isDriver = world.rank() == 0;
	std::string worker_name = "AlLib worker " + std::to_string(world.rank());
	log = (isDriver) ? start_log("AlLib driver") : start_log(worker_name);

	return 0;
}

int AlLib::unload() {

	return 0;
}

int AlLib::run(std::string task, Parameters & input, Parameters & output) {

	if (task.compare("kmeans") == 0) {

		uint32_t num_centers    = input.get_int("num_centers");
		uint32_t max_iterations = input.get_int("max_iterations");		// How many iteration of Lloyd's algorithm to use
		double epsilon          = input.get_double("epsilon");			// if all the centers change by Euclidean distance less
																		//     than epsilon, then we stop the iterations
		std::string init_mode   = input.get_string("init_mode");			// Number of initialization steps to use in kmeans||

		uint32_t init_steps     = input.get_int("init_steps");			// Which initialization method to use to choose
																		//     initial cluster center guesses
		uint64_t seed           = input.get_long("seed");					// Random seed used in driver and workers

		log->info("init_mode {}:", init_mode);
//		KMeans kmeans = new KMeans(num_centers, max_iterations, epsilon, init_mode, init_steps, seed);
		KMeans * kmeans = new KMeans(log, world, grid);
//		kmeans.set_log(log);
//		kmeans.set_world(world);
//		kmeans.set_peers(peers);
		kmeans->set_parameters(num_centers, max_iterations, epsilon, init_mode, init_steps, seed);
		kmeans->set_data_matrix(input.get_distmatrix("data"));
		kmeans->run(output);
	}
	else if (task.compare("truncated_svd") == 0) {

		SVD * svd = new SVD(log, world, grid);

		svd->set_rank(input.get_int("rank"));
		svd->set_data_matrix(input.get_distmatrix("data"));

		svd->run(output);
	}

	return 0;
}

}
