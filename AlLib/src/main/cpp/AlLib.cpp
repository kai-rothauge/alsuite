#include "AlLib.hpp"

using namespace allib;
using boost::mpi;

int AlLib::load(boost::mpi::environment & _env, boost::mpi::communicator & _world, boost::mpi::communicator & _peers) {

	env = _env;
	world = _world;
	peers = _peers;

	bool isDriver = world.rank() == 0;
	if isDriver
		std::shared_ptr<spdlog::logger> log = allib::start_log("AlLib driver");
	else
		std::shared_ptr<spdlog::logger> log = allib::start_log("AlLib worker");

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

	}
	else if (task.compare("svd") == 0) {
		SVD svd = new SVD(input, output, log);
	}

	return 0;
}

std::shared_ptr<spdlog::logger> allib::start_log(std::string name) {
	std::string logfile_name = name + ".log";

	std::shared_ptr<spdlog::logger> log;
	std::vector<spdlog::sink_ptr> sinks;
	sinks.push_back(std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>());
	sinks.push_back(std::make_shared<spdlog::sinks::simple_file_sink_st>(logfile_name));
	log = std::make_shared<spdlog::logger>(name, std::begin(sinks), std::end(sinks));
	log->flush_on(spdlog::level::info);
	log->set_level(spdlog::level::info); // only log stuff at or above info level, for production
	return log;
}
