#ifndef ALCHEMIST__ALCHEMIST_HPP
#define ALCHEMIST__ALCHEMIST_HPP

#include <omp.h>
#include <El.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include <boost/asio.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/tokenizer.hpp>
#include <boost/mpi.hpp>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <eigen3/Eigen/Dense>
//#include "arpackpp/arrssym.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"
// #include "spdlog/fmt/ostr.h"
#include <iostream>
#include <string>
#include <map>
#include <random>
#include "endian.hpp"
#include "Parameters.hpp"

#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#ifndef NDEBUG
#define ENSURE(x) assert(x)
#else
#define ENSURE(x) do { if(!(x)) { \
  fprintf(stderr, "FATAL: invariant violated: %s:%d: %s\n", __FILE__, __LINE__, #x); fflush(stderr); abort(); } while(0)
#endif

namespace alchemist {

namespace serialization = boost::serialization;
namespace mpi = boost::mpi;
using boost::format;

typedef El::Matrix<double> Matrix;
typedef El::AbstractDistMatrix<double> DistMatrix;
typedef uint32_t WorkerId;


//void kmeansPP(uint32_t seed, std::vector<Eigen::MatrixXd> points, std::vector<double> weights,
//    Eigen::MatrixXd & fitCenters, uint32_t maxIters);

struct MatrixHandle {
  uint32_t ID;

  template <typename Archive>
  void serialize(Archive &ar, const unsigned version) {
    ar & ID;
  }
};

struct WorkerInfo {
	std::string hostname;
	uint32_t port;

	template <typename Archive>
	void serialize(Archive &ar, const unsigned version) {
		ar & hostname;
		ar & port;
	}
};

inline bool operator < (const MatrixHandle & lhs, const MatrixHandle & rhs) {
  return lhs.ID < rhs.ID;
}

struct MatrixDescriptor {
	MatrixHandle handle;
	size_t num_rows;
	size_t num_cols;

	explicit MatrixDescriptor() :
		num_rows(0), num_cols(0) {
	}

	MatrixDescriptor(MatrixHandle handle, size_t num_rows, size_t num_cols) :
		handle(handle), num_rows(num_rows), num_cols(num_cols) {
	}

	template <typename Archive>
	void serialize(Archive &ar, const unsigned version) {
		ar & handle;
		ar & num_rows;
		ar & num_cols;
	}
};

inline bool exist_test (const std::string & name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

std::shared_ptr<spdlog::logger> start_log(std::string name);
}	// namespace alchemist

namespace fmt {
  // for displaying Eigen expressions. Note, if you include spdlog/fmt/ostr.h, this will be 
  // hidden by the ostream<< function for Eigen objects
  template <typename Formatter, typename Derived>
  inline void format_arg(Formatter &f,
      const char *&format_str, const Eigen::MatrixBase<Derived> &exp) {
    std::stringstream buf;
    buf << "Eigen matrix " << std::endl << exp;
    f.writer().write("{}", buf.str()); 
  }

  template <typename Formatter> 
  inline void format_arg(Formatter &f,
      const char *&format_str, const Eigen::Matrix<double, -1, -1> &exp) {
    std::stringstream buf;
    buf << "Eigen matrix " << std::endl << exp;
    f.writer().write("{}", buf.str()); 
  }

  // for displaying vectors
  template <typename T, typename A>
  inline void format_arg(BasicFormatter<char> &f, 
      const char *&format_str, const std::vector<T,A> &vec) {
    std::stringstream buf;
    buf << "Vector of length " << vec.size() << std::endl << "{";
    for(typename std::vector<T>::size_type pos=0; pos < vec.size()-1; ++pos) {
      buf << vec[pos] << "," << std::endl;
    }
    buf << vec[vec.size()-1] << "}";
    f.writer().write("{}", buf.str());
  }

  inline void format_arg(BasicFormatter<char> &f,
      const char *&format_str, const alchemist::MatrixHandle &handle) {
    f.writer().write("[{}]", handle.ID);
  }
}

namespace boost { namespace serialization {
  // to serialize Eigen Matrix objects
	template< class Archive,
						class S,
						int Rows_,
						int Cols_,
						int Ops_,
						int MaxRows_,
						int MaxCols_>
	inline void serialize(Archive & ar, 
		Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & matrix, 
		const unsigned int version)
	{
		int rows = matrix.rows();
		int cols = matrix.cols();
		ar & make_nvp("rows", rows);
		ar & make_nvp("cols", cols);    
		matrix.resize(rows, cols); // no-op if size does not change!

		// always save/load col-major
		for(int c = 0; c < cols; ++c)
			for(int r = 0; r < rows; ++r)
				ar & make_nvp("val", matrix(r,c));
	}
}} // namespace boost::serialization

BOOST_CLASS_EXPORT_KEY(alchemist::MatrixDescriptor);

#endif
