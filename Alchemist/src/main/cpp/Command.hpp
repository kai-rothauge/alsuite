#ifndef ALCHEMIST__COMMAND_HPP
#define ALCHEMIST__COMMAND_HPP

#include <omp.h>
#include <El.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include <boost/mpi.hpp>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <unistd.h>
#include <arpa/inet.h>
#include <eigen3/Eigen/Dense>
#include "spdlog/fmt/fmt.h"
// #include "spdlog/fmt/ostr.h"
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <map>
#include "data_stream.hpp"
#include "endian.hpp"
#include "Parameters.hpp"
#include "Executor.hpp"

namespace alchemist {

struct Executor;
struct Driver;
struct Worker;

struct Command {
	virtual ~Command() { }

	virtual void run(Worker * self) const = 0;

	template <typename Archive>
	void serialize(Archive &ar, const unsigned version) { }
};


struct LoadLibraryCommand : Command {
	void * sym;
	boost::mpi::environment & env;
	boost::mpi::communicator & world;
	boost::mpi::communicator & peers;

//	explicit LoadLibraryCommand() { }

	LoadLibraryCommand(void * _sym, boost::mpi::environment & _env, boost::mpi::communicator & _world,
			boost::mpi::communicator _peers) : sym(_sym), env(_env), world(_world), peers(_peers) { }

	void run(Worker *self) const;

	template <typename Archive>
	void serialize(Archive &ar, const unsigned version) {
		ar & serialization::base_object<Command>(*this);
	}
};

struct RunTaskCommand : Command {
	void * sym;
	std::string task_name;
	Parameters & input_parameters;
	Parameters & output_parameters;

//	explicit RunTaskCommand() { }

	RunTaskCommand(void * _sym, std::string _task_name, Parameters & _input_parameters, Parameters & _output_parameters) :
		sym(_sym), task_name(_task_name), input_parameters(_input_parameters), output_parameters(_output_parameters) {}

	void run(Worker *self) const {

		typedef int (*run_t)(std::string, const Parameters &, Parameters &);

		run_t run_f = (run_t) sym;
		run_f(task_name, input_parameters, output_parameters);

//		self->log->info("Finished call to {}::run", library_name);
	}

//	template <typename Archive>
//	void serialize(Archive &ar, const unsigned version) {
//		ar & serialization::base_object<Command>(*this);
//	}
};

struct HaltCommand : Command {
	void run(Worker *self) const {
//		self->shouldExit = true;
	}

	template <typename Archive>
	void serialize(Archive &ar, const unsigned version) {
		ar & serialization::base_object<Command>(*this);
	}
};


/*
struct ThinSVDCommand : Command {
  MatrixHandle mat;
  uint32_t whichFactors;
  uint32_t krank;
  MatrixHandle U;
  MatrixHandle S;
  MatrixHandle V;
  explicit ThinSVDCommand() {}
  ThinSVDCommand(MatrixHandle mat, uint32_t whichFactors, uint32_t krank,
      MatrixHandle U, MatrixHandle S, MatrixHandle V) :
    mat(mat), whichFactors(whichFactors), krank(krank), U(U), S(S), V(V) {}
  virtual void run(Worker *self) const;
  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_oject<Command>(*this);
    ar & mat;
    ar & whichFactors;
    ar & krank;
    ar & U;
    ar & S;
    ar & V;
  }
}
*/

struct TransposeCommand : Command {
  MatrixHandle origMat;
  MatrixHandle transposeMat;

  explicit TransposeCommand() {}

  TransposeCommand(MatrixHandle origMat, MatrixHandle transposeMat) :
    origMat(origMat), transposeMat(transposeMat) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & origMat;
    ar & transposeMat;
  }
};

struct KMeansCommand : Command {
  MatrixHandle origMat;
  uint32_t numCenters;
  uint32_t initSteps; // relevant in k-means|| only
  double changeThreshold; // stop when all centers change by Euclidean distance less than changeThreshold
  uint32_t method;
  uint64_t seed;
  MatrixHandle centersHandle;
  MatrixHandle assignmentsHandle;

  explicit KMeansCommand() {}

  KMeansCommand(MatrixHandle origMat, uint32_t numCenters, uint32_t method,
      uint32_t initSteps, double changeThreshold, uint64_t seed,
      MatrixHandle centersHandle, MatrixHandle assignmentsHandle) :
    origMat(origMat), numCenters(numCenters), method(method),
    initSteps(initSteps), changeThreshold(changeThreshold),
    seed(seed), centersHandle(centersHandle), assignmentsHandle(assignmentsHandle) {}

  virtual void run(Worker *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & origMat;
    ar & numCenters;
    ar & initSteps;
    ar & changeThreshold;
    ar & method;
    ar & seed,
    ar & centersHandle;
    ar & assignmentsHandle;
  }
};

struct TruncatedSVDCommand : Command {
  MatrixHandle mat;
  MatrixHandle UHandle;
  MatrixHandle SHandle;
  MatrixHandle VHandle;
  uint32_t k;

  explicit TruncatedSVDCommand() {}

  TruncatedSVDCommand(MatrixHandle mat, MatrixHandle UHandle,
      MatrixHandle SHandle, MatrixHandle VHandle, uint32_t k) :
    mat(mat), UHandle(UHandle), SHandle(SHandle), VHandle(VHandle),
    k(k) {}

  virtual void run(Executor *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & mat;
    ar & UHandle;
    ar & SHandle;
    ar & VHandle;
    ar & k;
  }
};

struct ThinSVDCommand : Command {
  MatrixHandle mat;
  MatrixHandle Uhandle;
  MatrixHandle Shandle;
  MatrixHandle Vhandle;

  explicit ThinSVDCommand() {}

  ThinSVDCommand(MatrixHandle mat, MatrixHandle Uhandle,
      MatrixHandle Shandle, MatrixHandle Vhandle) :
    mat(mat), Uhandle(Uhandle), Shandle(Shandle), Vhandle(Vhandle) {}

  virtual void run(Executor *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & mat;
    ar & Uhandle;
    ar & Shandle;
    ar & Vhandle;
  }
};

struct MatrixMulCommand : Command {
  MatrixHandle handle;
  MatrixHandle inputA;
  MatrixHandle inputB;

  explicit MatrixMulCommand() {}

  MatrixMulCommand(MatrixHandle dest, MatrixHandle A, MatrixHandle B) :
    handle(dest), inputA(A), inputB(B) {}

  virtual void run(Executor *self) const;

  template <typename Archive>
  void serialize(Archive & ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & handle;
    ar & inputA;
    ar & inputB;
  }
};

struct MatrixGetRowsCommand : Command {
  MatrixHandle handle;
  std::vector<WorkerId> layout;

  explicit MatrixGetRowsCommand() {}

  MatrixGetRowsCommand(MatrixHandle handle, std::vector<WorkerId> layout) :
    handle(handle), layout(layout) {}

  virtual void run(Executor * self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & handle;
    ar & layout;
  }
};

struct NewMatrixCommand : Command {
  MatrixDescriptor info;

  explicit NewMatrixCommand() {
  }

  NewMatrixCommand(const MatrixDescriptor &info) :
    info(info) {
  }

  virtual void run(Executor *self) const;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & serialization::base_object<Command>(*this);
    ar & info;
  }
};

} // namespace alchemist

BOOST_CLASS_EXPORT_KEY(alchemist::Command);
BOOST_CLASS_EXPORT_KEY(alchemist::LoadLibraryCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::RunTaskCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::HaltCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::NewMatrixCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::MatrixMulCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::MatrixGetRowsCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::ThinSVDCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::TransposeCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::KMeansCommand);
BOOST_CLASS_EXPORT_KEY(alchemist::TruncatedSVDCommand);

#endif
