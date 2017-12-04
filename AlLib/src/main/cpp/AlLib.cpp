#include "AlLib.hpp"

using namespace allib;
using boost::mpi;

extern "C"
int load(boost::mpi::environment & env, boost::mpi::communicator & world, boost::mpi::communicator & peers) {

	// Still lots to do here
//	El::Initialize();
	bool isDriver = world.rank() == 0;
	isDriver ? std::cerr << "I am library alLib driver" << std::endl : std::cerr << "I am library alLib worker" << std::endl;
	return 0;
}

extern "C"
int unload() {
	El::Finalize();
	return 0;
}

extern "C"
int run(std::string task, Parameters & input, Parameters & output, boost::mpi::environment & env,
		boost::mpi::communicator & world, boost::mpi::communicator & peers) {

	if (task.compare("kmeans") == 0) {
		kmeans(input, output, world, peers);
	}
	else if (task.compare("svd") == 0) {
		svd(input, output, world, peers);
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
