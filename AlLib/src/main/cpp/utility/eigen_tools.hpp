#include <eigen3/Eigen/Dense>
#include <stdio.h>
#include <string>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;


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

//  inline void format_arg(BasicFormatter<char> &f,
//      const char *&format_str, const alchemist::MatrixHandle &handle) {
//    f.writer().write("[{}]", handle.id);
//  }
}

namespace boost {

namespace serialization {
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
}

} // namespace boost::serialization
