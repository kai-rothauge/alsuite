#ifndef ALCHEMIST__LIBRARY_HPP
#define ALCHEMIST__LIBRARY_HPP

#include <El.hpp>
#include <boost/mpi.hpp>

namespace alchemist {

struct Library {

	Library(boost::mpi::communicator & _world, boost::mpi::communicator & _peers) : world(_world), peers(_peers), grid(El::mpi::Comm(peers)) { }

	virtual ~Library() { }

	boost::mpi::communicator & world;
	boost::mpi::communicator & peers;
	El::Grid grid;

	virtual int load(boost::mpi::communicator &, boost::mpi::communicator &) = 0;

	virtual int unload() = 0;

	virtual int run(std::string, Parameters &, Parameters &) = 0;
};

typedef Library * create_t(boost::mpi::communicator &, boost::mpi::communicator &);
typedef void destroy_t(Library *);

}

#endif
