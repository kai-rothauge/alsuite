#include "Executor.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace alchemist {


struct WorkerClientSendHandler {

	int sock;
	std::shared_ptr<spdlog::logger> log;
	short poll_events;
	std::vector<char> inbuf, outbuf;
	size_t inpos, outpos;
	const std::vector<uint64_t> & local_row_indices;
	const std::vector<double> & local_data;
	MatrixHandle handle;
	const size_t num_cols;

	// only set POLLOUT when have data to send
	// sends 0x3 code (uint32), then matrix handle (uint32), then row index (long = uint64_t)
	// local_data contains the rows of localRowIndices in order
	WorkerClientSendHandler(int sock, std::shared_ptr<spdlog::logger> log, MatrixHandle handle, size_t num_cols,
			const std::vector<uint64_t> & local_row_indices, const std::vector<double> & local_data) :
				sock(sock), log(log), poll_events(POLLIN), inbuf(16), outbuf(8 + num_cols * 8), inpos(0), outpos(0),
				local_row_indices(local_row_indices), local_data(local_data), handle(handle), num_cols(num_cols) { }

	~WorkerClientSendHandler() {
		close();
	}

	// note this is never used! It should be, to remove the client from the set of clients being polled once
	// the operation on that client is done
	bool is_closed() const {
		return sock == -1;
	}

	void close() {
		if (sock != -1) ::close(sock);
		sock = -1;
		poll_events = 0;
	}

	int handle_event(short revents) {
		mpi::communicator world;
		int rows_completed = 0;

		// Handle reads
		if(revents & POLLIN && poll_events & POLLIN) {
			while(!is_closed()) {
				int count = recv(sock, &inbuf[inpos], inbuf.size() - inpos, 0);
				//std::cerr << format("%s: read: sock=%s, inbuf=%s, inpos=%s, count=%s\n")
				//    % world.rank() % sock % inbuf.size() % inpos % count;
				if (count == 0) break;		// Means the other side has closed the socket
				else if( count == -1) {
					if(errno == EAGAIN) break;					// No more input available until next POLLIN
					else if (errno == EINTR) continue;			// Interrupted (e.g. by signal), so try again
					else if (errno == ECONNRESET) {
						close();
						break;
					} else abort();
				} else {
					ENSURE(count > 0);
					inpos += count;
					ENSURE(inpos <= inbuf.size());
					if (inpos >= 4) {
						char * data_ptr = &inbuf[0];
						uint32_t type_code = be32toh(*(uint32_t*) data_ptr);
						data_ptr += 4;
						if (type_code == 0x3 && inpos == inbuf.size()) {		// send row
							ENSURE(be32toh(*(uint32_t*) data_ptr) == handle.ID);
							data_ptr += 4;
							uint64_t row_index = htobe64(*(uint64_t*) data_ptr);
							data_ptr += 8;
							auto local_row_offset_iter = std::find(local_row_indices.begin(), local_row_indices.end(), row_index);
							ENSURE(local_row_offset_iter != local_row_indices.end());
							auto local_row_offset = local_row_offset_iter - local_row_indices.begin();
							*reinterpret_cast<uint64_t*>(&outbuf[0]) = be64toh(num_cols * 8);
							// treat the output as uint64_t[] instead of double[] to avoid type punning issues with be64toh
							auto invals = reinterpret_cast<const uint64_t*>(&local_data[num_cols * local_row_offset]);
							auto outvals = reinterpret_cast<uint64_t*>(&outbuf[8]);
							for(uint64_t i = 0; i < num_cols; ++i) outvals[i] = be64toh(invals[i]);
							inpos = 0;
							poll_events = POLLOUT; // after parsing the request, send the data
							break;
						}
					}
				}
			}
		}

		// Handle writes
		if (revents & POLLOUT && poll_events & POLLOUT) {
			// a la https://stackoverflow.com/questions/12170037/when-to-use-the-pollout-event-of-the-poll-c-function
			// and http://www.kegel.com/dkftpbench/nonblocking.html
			while (!is_closed()) {
				int count = write(sock, &outbuf[outpos], outbuf.size() - outpos);
				//std::cerr << format("%s: write: sock=%s, outbuf=%s, outpos=%s, count=%s\n")
				//    % world.rank() % sock % outbuf.size() % outpos % count;
				if (count == 0) break;
				else if (count == -1) {
					if (errno == EAGAIN) break;				// Out buffer is full for now, wait for next POLLOUT
					else if (errno == EINTR)	continue;		// Interrupted (e.g. by signal), so try again
					else if (errno == ECONNRESET) {
						close();
						break;
					} else abort();							// TODO
				}
				else {
					ENSURE(count > 0);
					outpos += count;
					ENSURE(outpos <= outbuf.size());
					if (outpos == outbuf.size()) { 			// After sending the row, wait for the next request
						rows_completed++;
						outpos = 0;
						poll_events = POLLIN;
						break;
					}
				}
			}
		}

		return rows_completed;
	}
};

struct WorkerClientReceiveHandler {

	int sock;
	short poll_events;
	std::vector<char> inbuf;
	size_t pos;
	DistMatrix * matrix;
	MatrixHandle handle;
	std::shared_ptr<spdlog::logger> log;

	WorkerClientReceiveHandler(int sock, std::shared_ptr<spdlog::logger> log, MatrixHandle handle, DistMatrix *matrix) :
		sock(sock), log(log), poll_events(POLLIN), inbuf(matrix->Width() * 8 + 24), pos(0), matrix(matrix), handle(handle) { }

	~WorkerClientReceiveHandler() {
		close();
	}

	bool is_closed() const {
		return sock == -1;
	}

	void close() {
		if(sock != -1) ::close(sock);
		//log->warn("Closed socket");
		sock = -1;
		poll_events = 0;
	}

	int handle_event(short revents) {
		mpi::communicator world;
		int rows_completed = 0;
		if(revents & POLLIN && poll_events & POLLIN) {
			while (!is_closed()) {
				//log->info("waiting on socket");
				int count = recv(sock, &inbuf[pos], inbuf.size() - pos, 0);
				//log->info("count of received bytes {}", count);
				if(count == 0) break;
				else if(count == -1) {
					if(errno == EAGAIN) break; 				// No more input available until next POLLIN
					else if(errno == EINTR) continue;
					else if(errno == ECONNRESET) {
						close();
						break;
					} else {
						log->warn("Something else happened to the connection");
						abort();								// TODO
					}
				} else {
					ENSURE(count > 0);
					pos += count;
					ENSURE(pos <= inbuf.size());
					if (pos >= 4) {
						char * data_ptr = &inbuf[0];
						uint32_t type_code = be32toh(*(uint32_t*) data_ptr);
						data_ptr += 4;
						if (type_code == 0x1 && pos == inbuf.size()) {		// add row
							size_t num_cols = matrix->Width();
							ENSURE(be32toh(*(uint32_t*) data_ptr) == handle.ID);
							data_ptr += 4;
							uint64_t row_index = htobe64(*(uint64_t*) data_ptr);
							data_ptr += 8;
							ENSURE(row_index < (size_t) matrix->Height());
							ENSURE(matrix->IsLocalRow(row_index));
							ENSURE(htobe64(*(uint64_t*) data_ptr) == num_cols * 8);
							data_ptr += 8;
							auto local_row_index = matrix->LocalRow(row_index);
							//log->info("Received row {} of matrix {}, writing to local row {}", rowIdx, handle.id, localRowIdx);
							for (size_t col_index = 0; col_index < num_cols; ++col_index) {
								double value = ntohd(*(uint64_t*) data_ptr);
								matrix->SetLocal(local_row_index, matrix->LocalCol(col_index), value); //LocalCal call should be unnecessary
								data_ptr += 8;
							}
							ENSURE(data_ptr == &inbuf[inbuf.size()]);
							//log->info("Successfully received row {} of matrix {}", rowIdx, handle.id);
							rows_completed++;
							pos = 0;
						} else if (type_code == 0x2) {
							//log->info("All the rows coming to me from one Spark executor have been received");
							/**struct sockaddr_storage addr;
							socklen_t len;
							char peername[255];
							int result = getpeername(sock, (struct sockaddr*)&addr, &len);
							ENSURE(result == 0);
							getnameinfo((struct sockaddr*)&addr, len, peername, 255, NULL, 0, 0);
							log->info("Received {} rows from {}", rowsCompleted, peername);
							**/
							pos = 0;
						}
					}
				}
			}
		}
		//log->info("returning from handling events");
		return rows_completed;
	}
};

Worker::Worker(boost::mpi::environment & _env, boost::mpi::communicator & _world, boost::mpi::communicator & _peers) :
    Executor(_env, _world, _peers), id(_world.rank() - 1), grid(El::mpi::Comm(peers)), listenSock(-1) {

    ENSURE(peers.rank() == world.rank() - 1);
    log = start_log(str(format("worker-%d") % world.rank()));
}

int Worker::run() {

	log->info("Started worker");
	log->info("Max number of OpenMP threads: {}", omp_get_max_threads());

	// create listening socket, bind to an available port, and get the port number
	ENSURE((listenSock = socket(AF_INET, SOCK_STREAM, 0)) != -1);
	sockaddr_in addr = {AF_INET};
	ENSURE(bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
	ENSURE(listen(listenSock, 1024) == 0);
	ENSURE(fcntl(listenSock, F_SETFL, O_NONBLOCK) != -1);
	socklen_t addrlen = sizeof(addr);
	ENSURE(getsockname(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen) == 0);
	ENSURE(addrlen == sizeof(addr));
	uint16_t port = be16toh(addr.sin_port);

	// transmit WorkerInfo to driver
	char hostname[256];
	ENSURE(gethostname(hostname, sizeof(hostname)) == 0);
	WorkerInfo info{hostname, port};
	world.send(0, 0, info);
	log->info("Listening for a connection at {}:{}", hostname, port);


	bool shouldExit = false;
	while(!shouldExit) {						// Handle commands until done
		uint32_t command_code;
		boost::mpi::broadcast(world, command_code, 0);
		switch(command_code) {
			// Shutdown
			case 0xFFFFFFFF:
				log->info("Shutting down");
				shouldExit = true;
				unload_libraries();
				break;

			case 0x0:
				// Do nothing
				break;

			case 0x1:
				// Do nothing
				break;

			case 0x2:
				// Do nothing
				break;

			case 0x3:
				// Do nothing
				break;

			case 0x4:
				log->info("load_library:");
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
				// Do nothing
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
		}
	}

	world.barrier();
	return EXIT_SUCCESS;
}

int Worker::load_library() {

	std::string args;
	boost::mpi::broadcast(world, args, 0);

	int status = Executor::load_library(args);

	world.barrier();

	return (status == 0) ? 0 : 1;
}

int Worker::run_task() {

	std::string args;
	boost::mpi::broadcast(world, args, 0);

	Parameters output_parameters = Parameters();

	int status = Executor::run_task(args, output_parameters);

	world.barrier();

	return (status == 0) ? 0 : 1;
}

int Worker::receive_new_matrix() {

	uint32_t handle_ID;
	uint64_t num_rows, num_cols;
	boost::mpi::broadcast(world, handle_ID, 0);
	boost::mpi::broadcast(world, num_rows, 0);
	boost::mpi::broadcast(world, num_cols, 0);

	MatrixHandle handle{handle_ID};

	log->info("Creating new distributed matrix");
	DistMatrix * matrix = new El::DistMatrix<double, El::MD, El::STAR>(num_rows, num_cols, grid);
	El::Zero(*matrix);
	ENSURE(matrices.insert(std::make_pair(handle, std::unique_ptr<DistMatrix>(matrix))).second);
	log->info("Created new distributed matrix");

	std::vector<uint64_t> rows_on_worker;
	log->info("Creating vector of local rows");
	rows_on_worker.reserve(num_rows);
	for (El::Int row_index = 0; row_index < num_rows; ++row_index)
		if (matrix->IsLocalRow(row_index)) rows_on_worker.push_back(row_index);

	for (int worker_index = 1; worker_index < world.size(); worker_index++) {
		if (world.rank() == worker_index) world.send(0, 0, rows_on_worker);
		world.barrier();
	}

	log->info("Starting to receive my assigned rows");
	receive_matrix_blocks(handle);
	log->info("Received all of my assigned rows");
	world.barrier();

	return 0;
}

int Worker::get_transpose() {

	uint32_t input_mat_ID, transpose_mat_ID;
	boost::mpi::broadcast(world, input_mat_ID, 0);
	boost::mpi::broadcast(world, transpose_mat_ID, 0);

	MatrixHandle input_handle{input_mat_ID};
	MatrixHandle transpose_handle{transpose_mat_ID};
	log->info("Constructing the transpose of matrix {}", input_handle);

	auto m = matrices[input_handle]->Height();
	auto n = matrices[input_handle]->Width();

	DistMatrix * transpose_A = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(n, m, grid);
	El::Zero(*transpose_A);
	ENSURE(matrices.insert(std::make_pair(transpose_handle, std::unique_ptr<DistMatrix>(transpose_A))).second);
	El::Transpose(*matrices[input_handle], *transpose_A);

	log->info("Finished transpose call");
	world.barrier();

	return 0;
}

int Worker::matrix_multiply() {

	uint32_t input_mat_A_ID, input_mat_B_ID, result_mat_ID;
	boost::mpi::broadcast(world, input_mat_A_ID, 0);
	boost::mpi::broadcast(world, input_mat_B_ID, 0);
	boost::mpi::broadcast(world, result_mat_ID, 0);

	MatrixHandle input_A_handle{input_mat_A_ID};
	MatrixHandle input_B_handle{input_mat_B_ID};
	MatrixHandle result_handle{result_mat_ID};

	log->info("Multiplying matrices {} and {}", input_A_handle.ID, input_B_handle.ID);

	auto m = matrices[input_A_handle]->Height();
	auto n = matrices[input_B_handle]->Width();

	DistMatrix * matrix = new El::DistMatrix<double, El::MC, El::MR, El::BLOCK>(m, n, grid);
	ENSURE(matrices.insert(std::make_pair(result_handle, std::unique_ptr<DistMatrix>(matrix))).second);
	El::Gemm(El::NORMAL, El::NORMAL, 1.0, *matrices[input_A_handle], *matrices[input_B_handle], 0.0, *matrix);

	log->info("Finished matrix multiplication call");
	world.barrier();

	return 0;
}

int Worker::get_matrix_rows() {

	uint32_t handle_ID;
	std::vector<uint32_t> layout;
	boost::mpi::broadcast(world, handle_ID, 0);
	boost::mpi::broadcast(world, layout, 0);

	MatrixHandle handle{handle_ID};

	uint64_t num_local_rows = std::count(layout.begin(), layout.end(), id);
	auto matrix = matrices[handle].get();
	uint64_t num_cols = matrix->Width();

	std::vector<uint64_t> local_row_indices; 			// Maps rows in the matrix to rows in the local storage
	std::vector<double> local_data(num_cols * num_local_rows);

	local_row_indices.reserve(num_local_rows);
	matrix->ReservePulls(num_cols * num_local_rows);
	for (uint64_t i = 0; local_row_indices.size() < num_local_rows; i++) {
		if (layout[i] == id) {
			local_row_indices.push_back(i);
			for (uint64_t col = 0; col < num_cols; col++) {
				matrix->QueuePull(i, col);
			}
		}
	}
	matrix->ProcessPullQueue(&local_data[0]);

	send_matrix_rows(handle, matrix->Width(), layout, local_row_indices, local_data);
	world.barrier();

	return 0;
}

int Worker::read_HDF5() {
	log->info("Worker::read_HDF5 not yet implemented");

	return 0;
}

int Worker::send_matrix_rows(MatrixHandle handle, size_t num_cols, const std::vector<WorkerId> & layout,
    const std::vector<uint64_t> & local_row_indices, const std::vector<double> & local_data) {

	auto num_local_rows = std::count(layout.begin(), layout.end(), this->id);
	std::vector<std::unique_ptr<WorkerClientSendHandler> > clients;
	std::vector<pollfd> pfds;

	while (num_local_rows > 0) {
		pfds.clear();
		for (auto it = clients.begin(); it != clients.end(); ) {
			const auto &client = *it;
			if (client->is_closed())
				it = clients.erase(it);
			else {
				pfds.push_back(pollfd{client->sock, client->poll_events});
				it++;
			}
		}
		pfds.push_back(pollfd{listenSock, POLLIN}); 		// Must be last entry

		int count = poll(&pfds[0], pfds.size(), -1);
		if (count == -1 && (errno == EAGAIN || errno == EINTR)) continue;
		ENSURE(count != -1);
		//log->info("Monitoring {} sockets (one is the listening socket)", pfds.size());
		for (size_t i = 0; i < pfds.size() && count > 0; ++i) {
			auto curSock = pfds[i].fd;
			auto revents = pfds[i].revents;
			if (revents != 0) {
				count--;
				if (curSock == listenSock) {
					ENSURE(revents == POLLIN);
					sockaddr_in addr;
					socklen_t addrlen = sizeof(addr);
					int clientSock = accept(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen);
					ENSURE(addrlen == sizeof(addr));
					ENSURE(fcntl(clientSock, F_SETFL, O_NONBLOCK) != -1);
					std::unique_ptr<WorkerClientSendHandler> client(new WorkerClientSendHandler(clientSock, log, handle, num_cols, local_row_indices, local_data));
					clients.push_back(std::move(client));
				} else {
					ENSURE(clients[i]->sock == curSock);
					num_local_rows -= clients[i]->handle_event(revents);
				}
			}
		}
	}
	log->info("Finished sending rows");

	return 0;
}

int Worker::receive_matrix_blocks(MatrixHandle handle) {

	std::vector<std::unique_ptr<WorkerClientReceiveHandler> > clients;
	std::vector<pollfd> pfds;
	uint64_t num_remaining_rows = matrices[handle].get()->LocalHeight();

	while (num_remaining_rows > 0) {
		//log->info("{} rows remaining", rowsLeft);
		pfds.clear();
		for (auto it = clients.begin(); it != clients.end(); ) {
			const auto & client = *it;
			if (client->is_closed())
				it = clients.erase(it);
			else {
				pfds.push_back(pollfd{client->sock, client->poll_events});
				it++;
			}
		}
		pfds.push_back(pollfd{listenSock, POLLIN});  	// Must be last entry
		//log->info("Pushed active clients to the polling list and added listening socket");
		int count = poll(&pfds[0], pfds.size(), -1);
		if (count == -1 && (errno == EAGAIN || errno == EINTR)) continue;
		ENSURE(count != -1);
		//log->info("Polled, now handling events");
		for (size_t i = 0; i < pfds.size() && count > 0; ++i) {
			auto curSock = pfds[i].fd;
			auto revents = pfds[i].revents;
			if (revents != 0) {
				count--;
				if (curSock == listenSock) {
					ENSURE(revents == POLLIN);
					sockaddr_in addr;
					socklen_t addrlen = sizeof(addr);
					int clientSock = accept(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen);
					ENSURE(addrlen == sizeof(addr));
					ENSURE(fcntl(clientSock, F_SETFL, O_NONBLOCK) != -1);
					std::unique_ptr<WorkerClientReceiveHandler> client(new WorkerClientReceiveHandler(clientSock, log, handle, matrices[handle].get()));
					clients.push_back(std::move(client));
					//log->info("Added new client");
				} else {
					ENSURE(clients[i]->sock == curSock);
					//log->info("Handling a client's events");
					num_remaining_rows -= clients[i]->handle_event(revents);
				}
			}
		}
	}

	return 0;
}

} // namespace alchemist
