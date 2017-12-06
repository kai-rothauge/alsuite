#ifndef ALLIB_HPP
#define ALLIB_HPP

#include <omp.h>
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
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <unistd.h>
#include <arpa/inet.h>
#include "include/Parameters.hpp"
#include "include/Library.hpp"
#include "utility/Logger.hpp"
#include "nla/nla.hpp"						// Include all NLA routines
#include "ml/ml.hpp"							// Include all ML/Data-mining routines

namespace allib {

typedef El::AbstractDistMatrix<double> DistMatrix;

struct AlLib : alchemist::Library, allib::Logger {

	boost::mpi::environment & env;
	boost::mpi::communicator & world;
	boost::mpi::communicator & peers;
	El::grid & grid;

	int load(boost::mpi::environment &, boost::mpi::communicator &, boost::mpi::communicator &, El::grid &);
	int unload();
	int run(std::string, Parameters &, Parameters &);
};

// Class factories
extern "C" Library* create() {
    return new AlLib;
}

extern "C" void destroy(Library* p) {
    delete p;
}

}

#endif // ALLIB_HPP
