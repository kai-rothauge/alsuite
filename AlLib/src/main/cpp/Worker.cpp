#include "alML.h"
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <poll.h>
#include "data_stream.h"
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "spdlog/spdlog.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace alML {

struct Worker {
  WorkerId id;
  mpi::communicator world;
  mpi::communicator peers;
  El::Grid grid;
  bool shouldExit;
  int listenSock;
  std::map<MatrixHandle, std::unique_ptr<DistMatrix>> matrices;
  std::shared_ptr<spdlog::logger> log;

  Worker(const mpi::communicator &world, const mpi::communicator &peers) :
      id(world.rank() - 1), world(world), peers(peers), grid(El::mpi::Comm(peers)),
      shouldExit(false), listenSock(-1) {
    ENSURE(peers.rank() == world.rank() - 1);
  }

  void receiveMatrixBlocks(MatrixHandle handle);
  void sendMatrixRows(MatrixHandle handle, size_t numCols, const std::vector<WorkerId> &layout,
      const std::vector<uint64_t> &localRowIndices, const std::vector<double> &localData);
  int main();
};

uint32_t updateAssignmentsAndCounts(MatrixXd const & dataMat, MatrixXd const & centers,
    uint32_t * clusterSizes, std::vector<uint32_t> & rowAssignments, double & objVal) {
  uint32_t numCenters = centers.rows();
  VectorXd distanceSq(numCenters);
  El::Int newAssignment;
  uint32_t numChanged = 0;
  objVal = 0.0;

  for(uint32_t idx = 0; idx < numCenters; ++idx)
    clusterSizes[idx] = 0;

  for(El::Int rowIdx = 0; rowIdx < dataMat.rows(); ++rowIdx) {
    for(uint32_t centerIdx = 0; centerIdx < numCenters; ++centerIdx)
      distanceSq[centerIdx] = (dataMat.row(rowIdx) - centers.row(centerIdx)).squaredNorm();
    objVal += distanceSq.minCoeff(&newAssignment);
    if (rowAssignments[rowIdx] != newAssignment)
      numChanged += 1;
    rowAssignments[rowIdx] = newAssignment;
    clusterSizes[rowAssignments[rowIdx]] += 1;
  }

  return numChanged;
}



void HaltCommand::run(Worker *self) const {
  self->shouldExit = true;
}

struct WorkerClientSendHandler {
  int sock;
  std::shared_ptr<spdlog::logger> log;
  short pollEvents;
  std::vector<char> inbuf;
  std::vector<char> outbuf;
  size_t inpos;
  size_t outpos;
  const std::vector<uint64_t> &localRowIndices;
  const std::vector<double> &localData;
  MatrixHandle handle;
  const size_t numCols;

  // only set POLLOUT when have data to send
  // sends 0x3 code (uint32), then matrix handle (uint32), then row index (long = uint64_t)
  // localData contains the rows of localRowIndices in order
  WorkerClientSendHandler(int sock, std::shared_ptr<spdlog::logger> log, MatrixHandle handle, size_t numCols, const std::vector<uint64_t> &localRowIndices, const std::vector<double> &localData) :
    sock(sock), log(log), pollEvents(POLLIN), inbuf(16), outbuf(8 + numCols * 8), inpos(0), outpos(0),
    localRowIndices(localRowIndices), localData(localData), handle(handle), numCols(numCols) {
  }

  ~WorkerClientSendHandler() {
    close();
  }

  // note this is never used! (it should be, to remove the client from the set of clients being polled once the operation on that client is done
  bool isClosed() const {
    return sock == -1;
  }

  void close() {
    if(sock != -1) ::close(sock);
    sock = -1;
    pollEvents = 0;
  }

  int handleEvent(short revents) {
    mpi::communicator world;
    int rowsCompleted = 0;

    // handle reads
    if(revents & POLLIN && pollEvents & POLLIN) {
      while(!isClosed()) {
        int count = recv(sock, &inbuf[inpos], inbuf.size() - inpos, 0);
        //std::cerr << format("%s: read: sock=%s, inbuf=%s, inpos=%s, count=%s\n")
        //    % world.rank() % sock % inbuf.size() % inpos % count;
        if (count == 0) {
          // means the other side has closed the socket
          break;
        } else if( count == -1) {
          if(errno == EAGAIN) {
            // no more input available until next POLLIN
            break;
          } else if(errno == EINTR) {
            // interrupted (e.g. by signal), so try again
            continue;
          } else if(errno == ECONNRESET) {
            close();
            break;
          } else {
            // TODO
            abort();
          }
        } else {
          ENSURE(count > 0);
          inpos += count;
          ENSURE(inpos <= inbuf.size());
          if(inpos >= 4) {
            char *dataPtr = &inbuf[0];
            uint32_t typeCode = be32toh(*(uint32_t*)dataPtr);
            dataPtr += 4;
            if(typeCode == 0x3 && inpos == inbuf.size()) {
              // sendRow
              ENSURE(be32toh(*(uint32_t*)dataPtr) == handle.id);
              dataPtr += 4;
              uint64_t rowIdx = htobe64(*(uint64_t*)dataPtr);
              dataPtr += 8;
              auto localRowOffsetIter = std::find(localRowIndices.begin(), localRowIndices.end(), rowIdx);
              ENSURE(localRowOffsetIter != localRowIndices.end());
              auto localRowOffset = localRowOffsetIter - localRowIndices.begin();
              *reinterpret_cast<uint64_t*>(&outbuf[0]) = be64toh(numCols * 8);
              // treat the output as uint64_t[] instead of double[] to avoid type punning issues with be64toh
              auto invals = reinterpret_cast<const uint64_t*>(&localData[numCols * localRowOffset]);
              auto outvals = reinterpret_cast<uint64_t*>(&outbuf[8]);
              for(uint64_t idx = 0; idx < numCols; ++idx) {
                outvals[idx] = be64toh(invals[idx]);
              }
              inpos = 0;
              pollEvents = POLLOUT; // after parsing the request, send the data
              break;
            }
          }
        }
      }
    }

    // handle writes
    if(revents & POLLOUT && pollEvents & POLLOUT) {
      // a la https://stackoverflow.com/questions/12170037/when-to-use-the-pollout-event-of-the-poll-c-function
      // and http://www.kegel.com/dkftpbench/nonblocking.html
      while(!isClosed()) {
        int count = write(sock, &outbuf[outpos], outbuf.size() - outpos);
        //std::cerr << format("%s: write: sock=%s, outbuf=%s, outpos=%s, count=%s\n")
        //    % world.rank() % sock % outbuf.size() % outpos % count;
        if (count == 0) {
          break;
        } else if(count == -1) {
          if(errno == EAGAIN) {
            // out buffer is full for now, wait for next POLLOUT
            break;
          } else if(errno == EINTR) {
            // interrupted (e.g. by signal), so try again
            continue;
          } else if(errno == ECONNRESET) {
            close();
            break;
          } else {
            // TODO
            abort();
          }
        } else {
          ENSURE(count > 0);
          outpos += count;
          ENSURE(outpos <= outbuf.size());
          if (outpos == outbuf.size()) { // after sending the row, wait for the next request
            rowsCompleted += 1;
            outpos = 0;
            pollEvents = POLLIN;
            break;
          }
        }
      }
    }

    return rowsCompleted;
  }
};

struct WorkerClientReceiveHandler {
  int sock;
  short pollEvents;
  std::vector<char> inbuf;
  size_t pos;
  DistMatrix *matrix;
  MatrixHandle handle;
  std::shared_ptr<spdlog::logger> log;

  WorkerClientReceiveHandler(int sock, std::shared_ptr<spdlog::logger> log, MatrixHandle handle, DistMatrix *matrix) :
      sock(sock), log(log), pollEvents(POLLIN), inbuf(matrix->Width() * 8 + 24),
      pos(0), matrix(matrix), handle(handle) {
  }

  ~WorkerClientReceiveHandler() {
    close();
  }

  bool isClosed() const {
    return sock == -1;
  }

  void close() {
    if(sock != -1) ::close(sock);
    //log->warn("Closed socket");
    sock = -1;
    pollEvents = 0;
  }

  int handleEvent(short revents) {
    mpi::communicator world;
    int rowsCompleted = 0;
    if(revents & POLLIN && pollEvents & POLLIN) {
      while(!isClosed()) {
        //log->info("waiting on socket");
        int count = recv(sock, &inbuf[pos], inbuf.size() - pos, 0);
        //log->info("count of received bytes {}", count);
        if(count == 0) {
          break;
        } else if(count == -1) {
          if(errno == EAGAIN) {
            // no more input available until next POLLIN
            //log->warn("EAGAIN encountered");
            break;
          } else if(errno == EINTR) {
            //log->warn("Connection interrupted");
            continue;
          } else if(errno == ECONNRESET) {
            //log->warn("Connection reset");
            close();
            break;
          } else {
            log->warn("Something else happened to the connection");
            // TODO
            abort();
          }
        } else {
          ENSURE(count > 0);
          pos += count;
          ENSURE(pos <= inbuf.size());
          if(pos >= 4) {
            char *dataPtr = &inbuf[0];
            uint32_t typeCode = be32toh(*(uint32_t*)dataPtr);
            dataPtr += 4;
            if(typeCode == 0x1 && pos == inbuf.size()) {
              // addRow
              size_t numCols = matrix->Width();
              ENSURE(be32toh(*(uint32_t*)dataPtr) == handle.id);
              dataPtr += 4;
              uint64_t rowIdx = htobe64(*(uint64_t*)dataPtr);
              dataPtr += 8;
              ENSURE(rowIdx < (size_t)matrix->Height());
              ENSURE(matrix->IsLocalRow(rowIdx));
              ENSURE(htobe64(*(uint64_t*)dataPtr) == numCols * 8);
              dataPtr += 8;
              auto localRowIdx = matrix->LocalRow(rowIdx);
              //log->info("Received row {} of matrix {}, writing to local row {}", rowIdx, handle.id, localRowIdx);
              for (size_t colIdx = 0; colIdx < numCols; ++colIdx) {
                double value = ntohd(*(uint64_t*)dataPtr);
                matrix->SetLocal(localRowIdx, matrix->LocalCol(colIdx), value); //LocalCal call should be unnecessary
                dataPtr += 8;
              }
              ENSURE(dataPtr == &inbuf[inbuf.size()]);
              //log->info("Successfully received row {} of matrix {}", rowIdx, handle.id);
              rowsCompleted++;
              pos = 0;
            } else if(typeCode == 0x2) {
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
    return rowsCompleted;
  }
};

void Worker::sendMatrixRows(MatrixHandle handle, size_t numCols, const std::vector<WorkerId> &layout,
    const std::vector<uint64_t> &localRowIndices, const std::vector<double> &localData) {
  auto numRowsFromMe = std::count(layout.begin(), layout.end(), this->id);
  std::vector<std::unique_ptr<WorkerClientSendHandler>> clients;
  std::vector<pollfd> pfds;
  while(numRowsFromMe > 0) {
    pfds.clear();
    for(auto it = clients.begin(); it != clients.end();) {
      const auto &client = *it;
      if(client->isClosed()) {
        it = clients.erase(it);
      } else {
        pfds.push_back(pollfd{client->sock, client->pollEvents});
        it++;
      }
    }
    pfds.push_back(pollfd{listenSock, POLLIN}); // must be last entry
    int count = poll(&pfds[0], pfds.size(), -1); 
    if(count == -1 && (errno == EAGAIN || errno == EINTR)) continue;
    ENSURE(count != -1);
    //log->info("Monitoring {} sockets (one is the listening socket)", pfds.size());
    for(size_t idx=0; idx < pfds.size() && count > 0; ++idx) {
      auto curSock = pfds[idx].fd;
      auto revents = pfds[idx].revents;
      if(revents != 0) {
        count--;
        if(curSock == listenSock) {
          ENSURE(revents == POLLIN);
          sockaddr_in addr;
          socklen_t addrlen = sizeof(addr);
          int clientSock = accept(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrlen);
          ENSURE(addrlen == sizeof(addr));
          ENSURE(fcntl(clientSock, F_SETFL, O_NONBLOCK) != -1);
          std::unique_ptr<WorkerClientSendHandler> client(new WorkerClientSendHandler(clientSock, log, handle, numCols, localRowIndices, localData));
          clients.push_back(std::move(client));
        } else {
          ENSURE(clients[idx]->sock == curSock);
          numRowsFromMe -= clients[idx]->handleEvent(revents);
        }
      }
    }
  }
  std::cerr << format("%s: finished sending rows\n") % world.rank();
}

void Worker::receiveMatrixBlocks(MatrixHandle handle) {
  std::vector<std::unique_ptr<WorkerClientReceiveHandler>> clients;
  std::vector<pollfd> pfds;
  uint64_t rowsLeft = matrices[handle].get()->LocalHeight(); 
  while(rowsLeft > 0) {
    //log->info("{} rows remaining", rowsLeft);
    pfds.clear();
    for(auto it = clients.begin(); it != clients.end();) {
      const auto &client = *it;
      if(client->isClosed()) {
        it = clients.erase(it);
      } else {
        pfds.push_back(pollfd{client->sock, client->pollEvents});
        it++;
      }
    }
    pfds.push_back(pollfd{listenSock, POLLIN});  // must be last entry
    //log->info("Pushed active clients to the polling list and added listening socket");
    int count = poll(&pfds[0], pfds.size(), -1);
    if(count == -1 && (errno == EAGAIN || errno == EINTR)) continue;
    ENSURE(count != -1);
    //log->info("Polled, now handling events");
    for(size_t idx = 0; idx < pfds.size() && count > 0; ++idx) {
      auto curSock = pfds[idx].fd;
      auto revents = pfds[idx].revents;
      if(revents != 0) {
        count--;
        if(curSock == listenSock) {
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
          ENSURE(clients[idx]->sock == curSock);
          //log->info("Handling a client's events");
          rowsLeft -= clients[idx]->handleEvent(revents);
        }
      }
    }
  }
}

int Worker::main() {
  // log to console as well as file (single-threaded logging)
  // TODO: allow to specify log directory, log level, etc.
  std::vector<spdlog::sink_ptr> sinks;
//  sinks.push_back(std::make_shared_ptr<spdlog::sinks::stdout_sink_st>()); // without ANSI color
  sinks.push_back(std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>()); // with ANSI color
  sinks.push_back(std::make_shared<spdlog::sinks::simple_file_sink_st>(str(format("rank-%d.log") % world.rank())));
  log = std::make_shared<spdlog::logger>( str(format("worker-%d") % world.rank()),
      std::begin(sinks), std::end(sinks));
  log->flush_on(spdlog::level::info); // change to warn for production
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

  // handle commands until done
  while(!shouldExit) {
    const Command *cmd = nullptr;
    mpi::broadcast(world, cmd, 0);
    cmd->run(this);
    delete cmd;
  }

  // synchronized exit
  world.barrier();
  return EXIT_SUCCESS;
}

int workerMain(const mpi::communicator &world, const mpi::communicator &peers) {
  return Worker(world, peers).main();
}

} // namespace alML
