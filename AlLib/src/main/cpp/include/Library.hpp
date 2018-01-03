#ifndef ALCHEMIST__LIBRARY_HPP
#define ALCHEMIST__LIBRARY_HPP

#include <El.hpp>
#include <boost/mpi.hpp>

namespace alchemist {

struct Library {

	Library(boost::mpi::communicator & _world) : world(_world) { }

	virtual ~Library() { }

	boost::mpi::communicator & world;
	El::Grid * grid;

	virtual int load() = 0;
	virtual int unload() = 0;
	virtual int run(std::string, Parameters &, Parameters &) = 0;

	int load_grid(El::Grid * _grid) { grid = _grid; }
};

typedef Library * create_t(boost::mpi::communicator &);
typedef void destroy_t(Library *);

}

#endif
