#ifndef ALCHEMIST__LOGGER_HPP
#define ALCHEMIST__LOGGER_HPP

#include "spdlog/spdlog.h"

namespace alchemist{

struct Logger {
	Logger() {};
	Logger(std::shared_ptr<spdlog::logger> & _log): log(_log) {};

	~Logger() {};

	std::shared_ptr<spdlog::logger> log;

	std::shared_ptr<spdlog::logger> start_log(std::string name) {
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
};

}

#endif
