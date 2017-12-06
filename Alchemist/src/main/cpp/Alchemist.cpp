#include "Alchemist.hpp"
#include "Executor.hpp"

int main(int argc, char *argv[]) {
	using namespace alchemist;
	boost::mpi::environment env;
	boost::mpi::communicator world;
	bool isDriver = world.rank() == 0;
	boost::mpi::communicator peers = world.split(isDriver ? 0 : 1);

	int status;

	if (isDriver) {
		std::shared_ptr<spdlog::logger> log = start_log("alchemist");

		log->info("Started Alchemist");

		char machine[255];
		char port[255];

		if (argc == 3) { // we are on a non-NERSC system, so passed in Spark driver machine name and port
			log->info("Non-NERSC system assumed");
			log->info("Connecting to Spark executor at {}:{}", argv[1], argv[2]);
			std::strcpy(machine, argv[1]);
			std::strcpy(port, argv[2]);
		}
		else { // assume we are on NERSC, so look in a specific location for a file containing the machine name and port
			char const* tmp = std::getenv("SPARK_WORKER_DIR");
			std::string sockPath;
			if (tmp == NULL) {
				log->info("Couldn't find the SPARK_WORKER_DIR variable");
				world.abort(1);
			} else {
				sockPath = std::string(tmp) + "/connection.info";
			}
			log->info("NERSC system assumed");
			log->info("Searching for connection information in file {}", sockPath);

			while(!exist_test(sockPath)) {
				boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
			}
			// now wait for a while for the connection file to be completely written, hopefully is enough time
			// TODO: need a more robust way of ensuring this is the case
			boost::this_thread::sleep_for(boost::chrono::milliseconds(500));

			std::string sockSpec;
			std::ifstream infile(sockPath);
			std::getline(infile, sockSpec);
			infile.close();
			boost::tokenizer<> tok(sockSpec);
			boost::tokenizer<>::iterator iter=tok.begin();
			std::string machineName = *iter;
			std::string portName = *(++iter);

			log->info("Connecting to Spark executor at {}:{}", machineName, portName);
			strcpy(machine, machineName.c_str());
			strcpy(port, portName.c_str());
		}

		boost::asio::ip::tcp::iostream stream(machine, port);
		ENSURE(stream);
		stream.rdbuf()->non_blocking(false);

		status = Driver(env, world, peers, stream, stream).run();

		log->info("Alchemist has exited");
	}
	else status = Worker(env, world, peers).run();

	return status;
}

std::shared_ptr<spdlog::logger> alchemist::start_log(std::string name) {
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

BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::MatrixDescriptor);

