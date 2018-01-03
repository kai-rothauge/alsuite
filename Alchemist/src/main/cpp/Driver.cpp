#include "Executor.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace alchemist {

using namespace El;

Driver::Driver(boost::mpi::environment & _env, boost::mpi::communicator & _world, boost::mpi::communicator & _peers,
		std::istream & is, std::ostream & os) :
    Executor(_env, _world, _peers), input(is), output(os), next_matrix_ID(42) {

	log = start_log("driver");
}

int Driver::run() {
	log->info("Started driver");
	log->info("Max number of OpenMP threads: {}", omp_get_max_threads());

	log->info("Registering workers");
	register_workers();

	bool shouldExit = false;
	while(!shouldExit) {
		uint32_t command_code = input.read_int();
		boost::mpi::broadcast(world, command_code, 0);
		switch (command_code) {
			// Shutdown
			case 0xFFFFFFFF:
				log->info("Shutting down");
				shouldExit = true;
				unload_libraries();
				output.write_int(0x1);
				output.flush();
				ENSURE(input.read_int() == 0x1);
				break;

			case 0x0:
				log->info("handshake:");
				handshake();
				break;

			case 0x1:
				log->info("receive_test_string:");
				receive_test_string();
				break;

			case 0x2:
				log->info("send_test_string:");
				send_test_string();
				break;

			case 0x3:
				log->info("send_worker_info:");
				send_worker_info();
				break;

			case 0x4:
				log->info("open_library:");
				load_library();
				break;

			case 0x5:
				log->info("run_task:");
				run_task();
				break;

			case 0x6:
				log->info("receive_new_matrix:");
				receive_new_matrix();
				break;

			case 0x7:
				log->info("get_matrix_dimensions:");
				get_matrix_dimensions();
				break;

			case 0x8:
				log->info("get_transpose:");
				get_transpose();
				break;

			case 0x9:
				log->info("matrix_multiply:");
				matrix_multiply();
				break;

			case 0x10:
				log->info("get_matrix_rows:");
				get_matrix_rows();
				break;

			case 0x11:
				log->info("read_HDF5:");
				read_HDF5();
				break;

			default:
				log->error("Unknown command code {#x}", command_code);
				abort();
		}
		if (!shouldExit) log->info("Waiting on next command");
	}

	// wait for workers to reach exit
	world.barrier();
	return EXIT_SUCCESS;
}

MatrixHandle Driver::register_matrix(size_t num_rows, size_t num_cols) {

	MatrixHandle handle{next_matrix_ID++};
	MatrixDescriptor info(handle, num_rows, num_cols);
	matrices.insert(std::make_pair(handle, info));

	return handle;
}

int Driver::handshake() {

	ENSURE(input.read_int() == 0xABCD);
	ENSURE(input.read_int() == 0x1);
	log->info("Successfully connected to Spark");
	output.write_int(0xDCBA);
	output.write_int(0x1);
	output.flush();

	return 0;
}

int Driver::receive_test_string() {

	std::string test_string = input.read_string();

	log->info("Test string from Alchemist.DriverClient:");
	log->info(test_string);

	return 0;
}

int Driver::send_test_string() {

	output.write_string("This is a test string from cpp alchemist");

	return 0;
}

int Driver::register_workers() {

	auto num_workers = world.size() - 1;
	workers.resize(num_workers);
	for(auto id = 0; id < num_workers; ++id)
		world.recv(id + 1, 0, workers[id]);
	log->info("{} workers ready", num_workers);

	return 0;
}

int Driver::send_worker_info() {

	log->info("Sending worker hostnames and ports to Spark");
	auto num_workers = world.size() - 1;
	output.write_int(num_workers);

	for(auto id = 0; id < num_workers; ++id) {
		output.write_string(workers[id].hostname);
		output.write_int(workers[id].port);
	}

	output.flush();

	return 0;
}

int Driver::process_input_parameters(Parameters & input_parameters) {

	// Nothing to do (at this point)

	return 0;
}

int Driver::process_output_parameters(Parameters & output_parameters) {

	unsigned distmatrices_count = 0;
	world.recv(1, 0, distmatrices_count);

	MatrixHandle handle;

	for (unsigned i = 0; i < distmatrices_count; i++) {
		std::string name;
		size_t num_rows, num_cols;
		world.recv(1, 0, name);
		world.recv(1, 0, num_rows);
		world.recv(1, 0, num_cols);

		handle = register_matrix(num_rows, num_cols);

		output_parameters.add_matrixhandle(name, handle.ID);

		boost::mpi::broadcast(world, handle, 0);
	}

	return 0;
}

int Driver::load_library() {

	std::string args = input.read_string();
	boost::mpi::broadcast(world, args, 0);

	int status = Executor::load_library(args);

	world.barrier();

	if (status != 0) {
		output.write_int(0x0);
		return 1;
	}

	output.write_int(0x1);

	return 0;
}

int Driver::run_task() {

	std::string args = input.read_string();
	boost::mpi::broadcast(world, args, 0);

	Parameters output_parameters = Parameters();

	int status = Executor::run_task(args, output_parameters);

	log->info("Output: {}", output_parameters.to_string());

	if (status != 0) {
		output.write_int(0x0);
		return 1;
	}

	output.write_int(0x1);
	output.write_string(output_parameters.to_string());

	return 0;
}

int Driver::receive_new_matrix() {

	uint64_t num_rows = input.read_long();
	uint64_t num_cols = input.read_long();

	MatrixHandle handle = register_matrix(num_rows, num_cols);
	log->info("Receiving new matrix {}, with dimensions {}x{}", handle.ID, num_rows, num_cols);

	boost::mpi::broadcast(world, handle.ID, 0);
	boost::mpi::broadcast(world, num_rows, 0);
	boost::mpi::broadcast(world, num_cols, 0);

	output.write_int(0x1);
	output.write_int(handle.ID);
	output.flush();
	world.barrier();

	// Tell Spark which worker expects each row
	std::vector<int> row_worker_assignments(num_rows, 0);
	std::vector<uint64_t> rows_on_worker;
	for (int worker_index = 1; worker_index < world.size(); worker_index++) {
		world.recv(worker_index, 0, rows_on_worker);
		world.barrier();
		for (auto row_index: rows_on_worker)
			row_worker_assignments[row_index] = worker_index;
	}

	log->info("Sending list of which worker each row should go to");
	output.write_int(0x1);
	for (auto worker_index: row_worker_assignments) {
		output.write_int(worker_index);
	}
	output.flush();

	log->info("Waiting for Spark to finish sending data to the workers");
	world.barrier();
	output.write_int(0x1);
	output.flush();
	log->info("Entire matrix has been received");

	return 0;
}

int Driver::get_matrix_dimensions() {

	MatrixHandle handle{input.read_int()};
	auto info = matrices[handle];
	output.write_int(0x1);
	output.write_long(info.num_rows);
	output.write_long(info.num_cols);
	output.flush();

	return 0;
}

int Driver::get_transpose() {

	MatrixHandle input_handle{input.read_int()};
	log->info("Constructing the transpose of matrix {}", input_handle.ID);

	auto num_rows = matrices[input_handle].num_cols;
	auto num_cols = matrices[input_handle].num_rows;

	MatrixHandle transpose_handle = register_matrix(num_rows, num_cols);

	boost::mpi::broadcast(world, input_handle.ID, 0);
	boost::mpi::broadcast(world, transpose_handle.ID, 0);

	world.barrier(); // wait for command to finish
	log->info("Finished transpose call");

	output.write_int(0x1);
	output.write_int(transpose_handle.ID);
	output.flush();

	return 0;
}

int Driver::matrix_multiply() {

	MatrixHandle input_A_handle{input.read_int()};
	MatrixHandle input_B_handle{input.read_int()};
	log->info("Multiplying matrices {} and {}", input_A_handle.ID, input_B_handle.ID);

	auto num_rows = matrices[input_A_handle].num_rows;
	auto num_cols = matrices[input_B_handle].num_cols;

	MatrixHandle result_handle = register_matrix(num_rows, num_cols);

	boost::mpi::broadcast(world, input_A_handle.ID, 0);
	boost::mpi::broadcast(world, input_B_handle.ID, 0);
	boost::mpi::broadcast(world, result_handle.ID, 0);

	world.barrier();			// Wait for the workers to finish
	log->info("Finished matrix multiplication call");

	output.write_int(0x1);
	output.write_int(result_handle.ID);
	output.flush();

	return 0;
}

int Driver::get_matrix_rows() {

	MatrixHandle handle{input.read_int()};
	uint64_t layout_length = input.read_long();
	std::vector<uint32_t> layout;
	layout.reserve(layout_length);

	for(uint64_t part = 0; part < layout_length; ++part) {
		layout.push_back(input.read_int());
	}
	log->info("Returning matrix {} to Spark", handle.ID);

	boost::mpi::broadcast(world, handle.ID, 0);
	boost::mpi::broadcast(world, layout, 0);

	// Tell Spark to start asking for rows
	output.write_int(0x1);

	output.flush();

	world.barrier();

	log->info("Finished fetching matrix rows");

	return 0;
}

int Driver::read_HDF5() {

	log->info("Driver::read_HDF5 not yet implemented");

	return 0;
}

} // namespace alchemist



