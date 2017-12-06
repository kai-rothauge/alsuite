#include "AlLib.hpp"

using namespace allib;
using boost::mpi;

int AlLib::load(boost::mpi::environment & _env, boost::mpi::communicator & _world,
		boost::mpi::communicator & _peers, El::Grid & _grid) {

	env = _env;
	world = _world;
	peers = _peers;
	grid = _grid;

	bool isDriver = world.rank() == 0;
	log = (isDriver) ? start_log("AlLib driver") : start_log("AlLib worker");

	return 0;
}

int AlLib::unload() {

	return 0;
}

int AlLib::run(std::string task, Parameters & input, Parameters & output) {

	if (task.compare("kmeans") == 0) {
		uint32_t data_handle    = input.get_matrix_handle("data");
		uint32_t num_centers    = input.get_int("k");
		uint32_t max_iterations = input.get_int("max_iterations");		// How many iteration of Lloyd's algorithm to use
		double epsilon          = input.get_double("epsilon");			// if all the centers change by Euclidean distance less
																		//     than epsilon, then we stop the iterations
		uint32_t init_mode      = input.get_string("init_mode");			// Number of initialization steps to use in kmeans||
		uint32_t init_steps     = input.get_int("init_steps");			// Which initialization method to use to choose
																		//     initial cluster center guesses
		uint64_t seed           = input.get_long("seed");					// Random seed used in driver and workers

		KMeans kmeans = new KMeans(num_centers, max_iterations, epsilon, init_mode, init_steps, seed);
		kmeans.set_log(log);
		kmeans.set_world(world);
		kmeans.set_peers(peers);
		kmeans.set_grid(grid);
		bool isDriver = world.rank() == 0;
		if (!isDriver)
			kmeans.set_data_matrix(std::dynamic_pointer_cast<DistMatrix>(input.get_ptr("data")));
		kmeans.run(output);
	}
	else if (task.compare("svd") == 0) {
		SVD svd = new SVD(input, output);
	}

	return 0;
}

