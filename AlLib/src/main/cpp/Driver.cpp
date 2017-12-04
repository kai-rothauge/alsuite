#include "alchemist.h"
#include "data_stream.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>
#include <map>
#include <random>
#include <sstream>
#include <boost/asio.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/tokenizer.hpp>
#include "arpackpp/arrssym.h"
#include "spdlog/spdlog.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace alML {

using namespace El;

struct Driver {
  mpi::communicator world;
  DataInputStream input;
  DataOutputStream output;
  std::vector<WorkerInfo> workers;
  std::map<MatrixHandle, MatrixDescriptor> matrices;
  uint32_t nextMatrixId;
  std::shared_ptr<spdlog::logger> log;

  Driver(const mpi::communicator &world, std::istream &is, std::ostream &os, std::shared_ptr<spdlog::logger> log);
  void issue(const Command &cmd);
  MatrixHandle registerMatrix(size_t numRows, size_t numCols);
  int main();

  void handle_newMatrix();
  void handle_matrixMul();
  void handle_matrixDims();
  void handle_computeThinSVD();
  void handle_getMatrixRows();
  void handle_getTranspose();
  void handle_kmeansClustering();
  void handle_truncatedSVD();
};

Driver::Driver(const mpi::communicator &world, std::istream &is, std::ostream &os, std::shared_ptr<spdlog::logger> log) :
    world(world), input(is), output(os), log(log), nextMatrixId(42) {
}

void Driver::issue(const Command &cmd) {
  const Command *cmdptr = &cmd;
  mpi::broadcast(world, cmdptr, 0);
}

MatrixHandle Driver::registerMatrix(size_t numRows, size_t numCols) {
  MatrixHandle handle{nextMatrixId++};
  MatrixDescriptor info(handle, numRows, numCols);
  matrices.insert(std::make_pair(handle, info));
  return handle;
}

int Driver::main() {
  //log to console as well as file (single-threaded logging)
  //TODO: allow to specify log directory, log level, etc.
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>());
  sinks.push_back(std::make_shared<spdlog::sinks::simple_file_sink_st>("driver.log"));
  log = std::make_shared<spdlog::logger>("driver", std::begin(sinks), std::end(sinks));
  log->flush_on(spdlog::level::info); // flush whenever warning or more critical message is logged
  log->set_level(spdlog::level::info); // only log stuff at or above info level, for production
  log->info("Started Driver");

  // get WorkerInfo
  auto numWorkers = world.size() - 1;
  workers.resize(numWorkers);
  for(auto id = 0; id < numWorkers; ++id) {
    world.recv(id + 1, 0, workers[id]);
  }
  log->info("{} workers ready, sending hostnames and ports to Spark", numWorkers);

  // handshake
  ENSURE(input.readInt() == 0xABCD);
  ENSURE(input.readInt() == 0x1);
  output.writeInt(0xDCBA);
  output.writeInt(0x1);
  output.writeInt(numWorkers);
  for(auto id = 0; id < numWorkers; ++id) {
    output.writeString(workers[id].hostname);
    output.writeInt(workers[id].port);
  }
  output.flush();

  bool shouldExit = false;
  while(!shouldExit) {
    uint32_t typeCode = input.readInt();
    log->info("Received code {:#x}", typeCode);

    switch(typeCode) {
      // shutdown
      case 0xFFFFFFFF:
        shouldExit = true;
        issue(HaltCommand());
        output.writeInt(0x1);
        output.flush();
        break;

      // new matrix
      case 0x1:
        handle_newMatrix();
        break;

      // matrix multiplication
      case 0x2:
        handle_matrixMul();
        break;

      // get matrix dimensions
      case 0x3:
        handle_matrixDims();
        break;

      // return matrix to Spark
      case 0x4:
        handle_getMatrixRows();
        break;

      case 0x5:
        handle_computeThinSVD();
        break;

      case 0x6:
        handle_getTranspose();
        break;

      case 0x7:
        handle_kmeansClustering();
        break;

      case 0x8:
        handle_truncatedSVD();
        break;

      default:
        log->error("Unknown typeCode {#x}", typeCode);
        abort();
    }
    log->info("Waiting on next command");
  }

  // wait for workers to reach exit
  world.barrier();
  return EXIT_SUCCESS;
}

inline bool exist_test (const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0); 
}

int driverMain(const mpi::communicator &world, int argc, char *argv[]) {
  //log to console as well as file (single-threaded logging)
  //TODO: allow to specify log directory, log level, etc.
  std::shared_ptr<spdlog::logger> log;
  std::vector<spdlog::sink_ptr> sinks;
  sinks.push_back(std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>());
  sinks.push_back(std::make_shared<spdlog::sinks::simple_file_sink_st>("driver.log"));
  log = std::make_shared<spdlog::logger>("driver", std::begin(sinks), std::end(sinks));
  //log->flush_on(spdlog::level::warn); // flush whenever warning or more critical message is logged
  //log->set_level(spdlog::level::info); // only log stuff at or above info level, for production
  log->flush_on(spdlog::level::info); // flush always, for debugging
  log->info("Started Driver");
  log->info("Max number of OpenMP threads: {}", omp_get_max_threads());

  char machine[255];
  char port[255];

  if (argc == 3) { // we are on a non-NERSC system, so passed in Spark driver machine name and port
      log->info("Non-NERSC system assumed");
      log->info("Connecting to Spark executor at {}:{}", argv[1], argv[2]);
      std::strcpy(machine, argv[1]);
      std::strcpy(port, argv[2]);
  } else { // assume we are on NERSC, so look in a specific location for a file containing the machine name and port
      char const* tmp = std::getenv("SPARK_WORKER_DIR");
      std::string sockPath;
      if (tmp == NULL) {
          log->info("Couldn't find the SPARK_WORKER_DIR variable");
          world.abort(1);
      } else {
        sockPath = std::string(tmp) + "/connection.info";
      }
      log->info("NERSC system assumed");
      log->info("Searching for connection information in file {}", sockPath);

      while(!exist_test(sockPath)) {
          boost::this_thread::sleep_for(boost::chrono::milliseconds(50));
      }
      // now wait for a while for the connection file to be completely written, hopefully is enough time
      // TODO: need a more robust way of ensuring this is the case
      boost::this_thread::sleep_for(boost::chrono::milliseconds(500));

      std::string sockSpec;
      std::ifstream infile(sockPath);
      std::getline(infile, sockSpec);
      infile.close();
      boost::tokenizer<> tok(sockSpec);
      boost::tokenizer<>::iterator iter=tok.begin();
      std::string machineName = *iter;
      std::string portName = *(++iter);

      log->info("Connecting to Spark executor at {}:{}", machineName, portName);
      strcpy(machine, machineName.c_str());
      strcpy(port, portName.c_str());
  }

  using boost::asio::ip::tcp;
  boost::asio::ip::tcp::iostream stream(machine, port);
  ENSURE(stream);
  stream.rdbuf()->non_blocking(false);
  auto result = Driver(world, stream, stream, log).main();
  return result;
}

} // namespace alML
