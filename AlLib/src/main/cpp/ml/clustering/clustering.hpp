#ifndef ALLIB__CLUSTERING_HPP
#define ALLIB__CLUSTERING_HPP

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
#include <boost/mpi.hpp>
#include "../../utility/Logger.hpp"
#include "../../include/Parameters.hpp"

namespace allib {

typedef El::AbstractDistMatrix<double> DistMatrix;

using Eigen::MatrixXd;
using Eigen::VectorXd;

using alchemist::Parameters;

class Clustering : Logger {
public:
	Clustering(std::shared_ptr<spdlog::logger> & _log, boost::mpi::communicator & _world, boost::mpi::communicator & _peers) :
		Logger(_log), world(_world), peers(_peers), grid(El::mpi::Comm(peers)) {}

	~Clustering() {}

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

	virtual int initialize() = 0;
	virtual int train(Parameters & output) = 0;
};

class KMeans : public Clustering {
public:
//	KMeans();
	KMeans(std::shared_ptr<spdlog::logger> & _log, boost::mpi::communicator & _world, boost::mpi::communicator & _peers) :
		Clustering(_log, _world, _peers) {}

//	KMeans(uint32_t, uint32_t, double, std::string, uint32_t, uint64_t);

	void set_parameters(uint32_t _num_centers, uint32_t _max_iterations, double _epsilon, std::string _init_mode,
			uint32_t _init_steps, uint64_t _seed);

	uint32_t get_num_centers();
	void set_num_centers(uint32_t);

	uint32_t get_max_iterations();
	void set_max_iterations(uint32_t);

	std::string get_init_mode();
	void set_init_mode(std::string);

	uint32_t get_init_steps();
	void set_init_steps(uint32_t);

	double get_epsilon();
	void set_epsilon(double);

	uint64_t get_seed();
	void set_seed(uint64_t);

	void set_data_matrix(DistMatrix * _data);

	int initialize();
	int train(Parameters & output);
	int run(Parameters & output);

private:
	uint32_t num_centers;
	uint32_t max_iterations;					// How many iteration of Lloyd's algorithm to use at most
	double epsilon;							// If all the centers change by Euclidean distance less
											//     than epsilon, then we stop the iterations
	std::string init_mode;					// Number of initialization steps to use in kmeans||
	uint32_t init_steps;						// Which initialization method to use to choose
											//     initial cluster center guesses
	uint64_t seed;							// Random seed used in driver and workers

	DistMatrix * data_matrix;

	int initialize_random();
	int initialize_parallel(DistMatrix const *, MatrixXd const &, uint32_t, MatrixXd &);

	uint32_t update_assignments_and_counts(MatrixXd const &, MatrixXd const &,
	    uint32_t *, std::vector<uint32_t> &, double &);

	int kmeansPP(std::vector<MatrixXd>, std::vector<double>, MatrixXd &);
};

}

#endif // ALLIB__CLUSTERING_HPP
