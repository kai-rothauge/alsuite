#ifndef ALLIB__SVD_HPP
#define ALLIB__SVD_HPP

#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <poll.h>
#include <thread>
#include <El.hpp>
#include <stdio.h>
#include <string>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include <boost/random.hpp>
#include "../utility/Logger.hpp"
#include "../include/Parameters.hpp"

namespace allib {

typedef El::AbstractDistMatrix<double> DistMatrix;

using alchemist::Parameters;

class SVD : Logger {
public:

	SVD(std::shared_ptr<spdlog::logger> & _log, boost::mpi::communicator & _world, boost::mpi::communicator & _peers) :
			Logger(_log), world(_world), peers(_peers), grid(El::mpi::Comm(peers)) {}

	boost::mpi::communicator & world;
	boost::mpi::communicator & peers;
	El::Grid grid;

	void set_log(std::shared_ptr<spdlog::logger> & _log) {
		log = _log;
	}

	void set_world(boost::mpi::communicator & _world) {
		world = _world;
	}

	void set_peers(boost::mpi::communicator & _peers) {
		peers = _peers;
	}

	int run(Parameters & output);
};

}

#endif
