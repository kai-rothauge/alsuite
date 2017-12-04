#ifndef ALCHEMIST__EXECUTOR_HPP
#define ALCHEMIST__EXECUTOR_HPP

#include "data_stream.hpp"

namespace alchemist {

struct Executor {
	boost::mpi::environment & env;
	boost::mpi::communicator & world;
	boost::mpi::communicator & peers;

	std::shared_ptr<spdlog::logger> log;

	std::map<std::string, void *> libraries;

	Executor(boost::mpi::environment & _env, boost::mpi::communicator & _world, boost::mpi::communicator & _peers) :
		env(_env), world(_world), peers(_peers) {}

	virtual int run() = 0;

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
	void issue(const Command &cmd);
	MatrixHandle register_matrix(size_t num_rows, size_t num_cols);

	int run();

	int handshake();

	int receive_test_string();
	int send_test_string();

	int register_workers();
	int send_worker_info();

	int load_library();
	int run_task();

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
	El::Grid grid;
	std::map<MatrixHandle, std::unique_ptr<DistMatrix> > matrices;

	Worker(boost::mpi::environment & _env, boost::mpi::communicator & _world, boost::mpi::communicator & _peers);

	int run();

	int load_library();
	int run_task();

	int receive_new_matrix();
	int get_transpose();
	int matrix_multiply();
	int get_matrix_rows();
	int read_HDF5();

	int receive_matrix_blocks(MatrixHandle handle);
	int send_matrix_rows(MatrixHandle handle, size_t num_cols, const std::vector<WorkerId> & layout, const std::vector<uint64_t> & local_row_indices, const std::vector<double> & local_data);
};

}

#endif
