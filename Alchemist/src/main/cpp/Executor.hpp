#ifndef ALCHEMIST__EXECUTOR_HPP
#define ALCHEMIST__EXECUTOR_HPP

#include <dlfcn.h>
#include "data_stream.hpp"
#include "Library.hpp"
#include "utility/Logger.hpp"

namespace alchemist {

struct LibraryInfo {

	LibraryInfo(std::string _name, std::string _path, void * _lib_ptr, Library * _lib) :
		name(_name), path(_path), lib_ptr(_lib_ptr), lib(_lib) {}

	std::string name;
	std::string path;

	void * lib_ptr;

	Library * lib;
};

struct Executor : Logger {
	boost::mpi::environment & env;
	boost::mpi::communicator & world;
	boost::mpi::communicator & peers;

	std::map<std::string, LibraryInfo> libraries;

	Executor(boost::mpi::environment & _env, boost::mpi::communicator & _world, boost::mpi::communicator & _peers) :
		Logger(), env(_env), world(_world), peers(_peers) {}

	virtual int run() = 0;

	virtual int process_input_parameters(Parameters &) = 0;
	virtual int process_output_parameters(Parameters &) = 0;

	int load_library(std::string);
	int run_task(std::string, Parameters &);
	int unload_libraries();

//	int receive_new_matrix();
//	int get_transpose();
//	int matrix_multiply();
//	int get_matrix_rows();
//	int read_HDF5();

	void deserialize_parameters(std::string, Parameters &);
	std::string serialize_parameters(const Parameters &) const;
};

struct Driver : Executor {

	DataInputStream input;
	DataOutputStream output;

	std::vector<WorkerInfo> workers;
	std::map<MatrixHandle, MatrixDescriptor> matrices;
	uint32_t next_matrix_ID;

	Driver(boost::mpi::environment & _env, boost::mpi::communicator & _world, boost::mpi::communicator & _peers,
			std::istream & is, std::ostream & os);

	MatrixHandle register_matrix(size_t num_rows, size_t num_cols);

	int run();

	int handshake();

	int receive_test_string();
	int send_test_string();

	int register_workers();
	int send_worker_info();

	int load_library();
	int run_task();

	int process_input_parameters(Parameters &);
	int process_output_parameters(Parameters &);

	int receive_new_matrix();
	int get_matrix_dimensions();
	int get_transpose();
	int matrix_multiply();
	int get_matrix_rows();
	int read_HDF5();
};

struct Worker : Executor {

	WorkerId id;
	int listenSock;
	std::map<MatrixHandle, std::shared_ptr<DistMatrix> > matrices;

	El::Grid grid;

	Worker(boost::mpi::environment &, boost::mpi::communicator &, boost::mpi::communicator &);

	int run();

	int load_library();
	int run_task();

	int process_input_parameters(Parameters &);
	int process_output_parameters(Parameters &);

	int receive_new_matrix();
	int get_transpose();
	int matrix_multiply();
	int get_matrix_rows();
	int read_HDF5();

	int receive_matrix_blocks(MatrixHandle handle);
	int send_matrix_rows(MatrixHandle handle, size_t num_cols, const std::vector<WorkerId> & layout,
			const std::vector<uint64_t> & local_row_indices, const std::vector<double> & local_data);
};

}

#endif
