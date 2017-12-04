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
#include <boost/mpi.hpp>
#include "spdlog/spdlog.h"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <unistd.h>
#include <arpa/inet.h>
#include "Parameters.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;


//#include "src/utility/utility.hpp"				/** Include all the utility functions */
//#include "src/base/base.hpp"						/** Include the base of the library */
//#include "src/algorithms/algorithms.hpp"			/** Include all basic algorithms provided */
//#include "src/nla/nla.hpp"						/** Include all NLA routines */
//#include "src/ml/ml.hpp"							/** Include all ML/Data-mining routines */

namespace allib {

extern "C"
int load(boost::mpi::environment &, boost::mpi::communicator &, boost::mpi::communicator &);

extern "C"
int unload();

extern "C"
int run(std::string, Parameters &, Parameters &, boost::mpi::environment &,
		boost::mpi::communicator &, boost::mpi::communicator &);

int kmeans(Parameters &, Parameters &, boost::mpi::communicator &, boost::mpi::communicator &);

std::shared_ptr<spdlog::logger> start_log(std::string name);
}

#endif // ALLIB_HPP
