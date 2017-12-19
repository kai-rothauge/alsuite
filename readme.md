Alchemist is a framework for easily and efficiently calling MPI-based codes from Apache Spark. 

# Alchemist Framework Structure

The Alchemist framework has a modular structure in order to achieve flexibility and ease of use. We distinguish between three "layers":
* The core Alchemist system
* The library layer, consisting of one or more MPI-based libraries, as well as two interfaces for each of these libraries, one for bridging Apache Spark and Alchemist, and another one for bridging the MPI library and Alchemist.
* The application layer, consisting of some Spark-based code that wishes to call one or more MPI libraries.

See below for a schematic representation in the case of one MPI library:

![Alchemist Structure](https://github.com/kai-rothauge/alsuite/images/alchemist_framework.png)

We also distinguish between the Spark side and the MPI side: 
* The Spark (or Scala) side consists of the Spark application, the Spark-Alchemist interface for each library, and the Alchemist subsystem that takes in a Spark-based data structure. This part of the core Alchemist code is written in Scala, and it is expected that the Spark application and interface are written in Scala as well, although this requirement may be eased at some point.
* The MPI (or C++) side consists of the MPI libraries, the library-Alchemist interface for each library, and the Alchemist subsystem that takes in the data from the Spark side and creates the distributed data structures that are needed by the libraries. This subsystem is written in C++ and it is expected that the library and its interface are also written in C++, although this requirement may also be eased at some point. 

For now we require that the MPI libraries use the Elemental library and Boost MPI.

# Library Interfaces

More details to follow.

# AlSuite

AlSuite is a sample framework that illustrates the use of Alchemist using a small (and very limited) MPI library AlLib, and a sample Spark-based application, AlTest, that the user can use to evaluate the performance of the methods contained in AlLib compared to the equivalent methods in Spark.

## Sample MPI library: AlLib

For now AlLib contains just two procedures, K-Means (as an example of a supervised machine learning procedure), and truncated SVD (as an example of a numerical linear algebra routine). 

## Sample Spark Application: AlTest

AlTest provides the user with a small testing application that illustrates the difference in performance between Spark and AlLib for the procedures implemented in AlLib.

# Dependencies

The following dependencies are required by AlLib and the MPI side of Alchemist:

## Alchemist Dependencies

The MPI side of the core Alchemist code requires the following supporting libraries:
* Elemental: For distributing the matrices between Alchemist processes and distributed linear algebra.
* SPDLog: For thread-safe logging during execution.

## AlLib Dependencies

AlLib requires the following supporting libraries:
* Elemental: For distributing the matrices between Alchemist processes and distributed linear algebra.
* SPDLog: For thread-safe logging during execution.
* Eigen3 -- used for local matrix manipulations (more convenient interface than Elemental)
* Arpack-ng -- for the computation of truncated SVDs
* Arpackpp -- very convenient C++ interface to Arpack-ng

See below for installation instructions for each of these libraries.

# Installation instructions

## MacOS 10.12

### Prerequisites

The following prerequisites are needed by the Alchemist framework. Assuming that the XCode command line tools, Homebrew, and Spark have been installed:

```
brew install gcc
brew install make --with-default-names
brew install cmake
brew install boost-mpi
brew install sbt
```

### Clone the AlSuite repo
```
export ALSUITE_ROOT=(/desired/path/to/AlSuite/parent/directory)
cd $ALSUITE_ROOT
git clone https://github.com/kai-rothauge/alsuite.git			# This will change at some point to Alex's repo
export ALSUITE_PATH=$ALSUITE_ROOT/AlSuite
```

### Install dependencies

If some or all of the dependencies listed above have not yet been installed on the system, follow the instructions below. Elemental and SPDLog are needed by Alchemist, AlLib additionally requires Eigen3, Arpack-ng and Arpackpp.

#### Install Elemental
```
export ELEMENTAL_PATH=(/desired/path/to/Elemental/directory)
git clone https://github.com/elemental/Elemental.git
cd Elemental
git checkout 0.87
mkdir build
cd build
CC=gcc-7 CXX=g++-7 FC=gfortran-7 cmake -DCMAKE_BUILD_TYPE=Release -DEL_IGNORE_OSX_GCC_ALIGNMENT_PROBLEM=ON -DCMAKE_INSTALL_PREFIX=$ELEMENTAL_PATH ..
nice make -j8
make install
cd ../..
rm -rf Elemental
```

#### Install SPDLog
```
export SPDLOG_PATH=(/desired/path/to/SPDLog/directory)
git clone https://github.com/gabime/spdlog.git
cd spdlog
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$SPDLOG_PATH ..
make install
cd ../..
rm -rf spdlog
```

#### Install Eigen3
```
export EIGEN3_PATH=(/desired/path/to/Eigen3/directory)
curl -L -O http://bitbucket.org/eigen/eigen/get/3.3.4.zip
unzip 3.3.4.zip
rm 3.3.4.zip
cd eigen-eigen-************
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$EIGEN3_PATH ..
make install
cd ../..
rm -rf eigen-eigen-************
```

#### Install Arpack-ng
```
export ARPACK_PATH=(/desired/path/to/Arpack-ng/directory)
git clone https://github.com/opencollab/arpack-ng.git
cd arpack-ng
mkdir build
cd build
CC=gcc-7 FC=gfortran-7 cmake -DMPI=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$ARPACK_PATH ..
nice make -j8
make install
cd ../..
rm -rf arpack-ng
```

#### Install Arpackpp 
```
export ARPACK_PATH=(/desired/path/to/Arpack-ng/directory)
git clone https://github.com/m-reuter/arpackpp.git
cd arpackpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$ARPACK_PATH ..
make install
cd ../..
rm -rf arpackpp
```

Note that the installation paths to Arpack-ng and Arpackpp should be the same.

### Update Makefiles

The following makefiles need to be updated:
* In ``$(ALSUITE_PATH)\Alchemist\Makefile``, set ``ELEMENTAL_PATH`` and ``SPDLOG_PATH`` (at the top of the file) to the relevant directories.
* In ``$(ALSUITE_PATH)\AlLib\Makefile``, set ``ELEMENTAL_PATH``, ``SPDLOG_PATH``, ``EIGEN3_PATH`` and ``ARPACK_PATH`` (at the top of the file) to the relevant directories.

Alternatively, one can add the above paths to the bash profile.

# Building AlSuite

Assuming the above libraries have been installed (see below for instructions), AlSuite can be built on a MacBook as follows:

```
export ALSUITE_PATH=/path/to/AlSuite/directory
export PATH=$PATH:/path/to/spark-bin/directory 
export TMPDIR=/tmp 				# avoid a Mac specific issue with tmpdir length
cd $ALSUITE_PATH
./build_alsuite.sh
```

It's probably a good idea to add the above export statements to the bash profile.

# Running AlTest

AlTest can be configured in the file ``$(ALSUITE_ROOT)\AlTest\config\config.py``, where the user can choose which routines or procedures to test, as well as associated configurations and settings.

AlTest can then be run as follows:

```
cd $ALSUITE_PATH
./start_altest.sh
```
