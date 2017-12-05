#ifndef ALCHEMIST__LIBRARY_HPP
#define ALCHEMIST__LIBRARY_HPP

#include <boost/mpi.hpp>

namespace alchemist {

struct Library {

	Library() { }

	virtual ~Library() { }

	virtual int load(boost::mpi::environment &, boost::mpi::communicator &, boost::mpi::communicator &) = 0;

	virtual int unload() = 0;

	virtual int run(std::string, Parameters &, Parameters &) = 0;
};

typedef Library * open_t();
typedef void close_t(Library *);

}

#endif
