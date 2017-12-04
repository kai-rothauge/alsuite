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
		KMeans kmeans = new KMeans(input, output);

	}
	else if (task.compare("svd") == 0) {
		SVD svd = new SVD(input, output, world, peers);
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
