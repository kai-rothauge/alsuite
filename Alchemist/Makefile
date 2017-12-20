ALCHEMIST_PATH ?= $(ALSUITE_PATH)/Alchemist
ELEMENTAL_PATH ?= $(HOME)/Software/Elemental
EIGEN3_PATH ?= $(HOME)/Software/Eigen3
SPDLOG_PATH ?= $(HOME)/Software/SPDLog

include $(ELEMENTAL_PATH)/conf/ElVars

SRC_PATH = $(ALCHEMIST_PATH)/src/main/cpp
TARGET_PATH = $(ALCHEMIST_PATH)/target

# put libEl's CXXFLAGS in front so ours can override it
CXXFLAGS += $(EL_COMPILE_FLAGS) -fdiagnostics-color=always
#CXXFLAGS += -Wall
CXXFLAGS += -Wno-unused -Wno-reorder -std=c++14 -fext-numeric-literals -fopenmp
CXXFLAGS += "-I$(SRC_PATH)/utility"
CXXFLAGS += "-I$(ELEMENTAL_PATH)/include" "-I$(EIGEN3_PATH)/include" "-I$(SPDLOG_PATH)/include"

LDFLAGS += "-L$(EL_LIB)" "-Wl,-rpath,$(EL_LIB)" $(EL_LIBS)
ifeq ($(shell uname), Darwin)
LDFLAGS += -lboost_serialization-mt -lboost_mpi-mt -lboost_system-mt -lboost_thread-mt
else
LDFLAGS += -lboost_serialization -lboost_mpi -lboost_system -lboost_thread 
endif
LDFLAGS += -Wl,-rpath,$(SPDLOG_PATH)/lib

# poor heuristic to check if on Cori, and fix a linkage issue that shouldn't be occurring
ifeq ($(shell uname), Linux)
LDFLAGS += -Wl,-rpath,/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64_lin/ -L/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64_lin/ -lirc
endif

ifeq ($(shell uname), Darwin)
LDFLAGS += -lmpi
endif

# some environments (e.g. Cori) require this to find boost
ifdef BOOST_DIR
  CXXFLAGS += "-I$(BOOST_DIR)/include"
  LDFLAGS += "-L$(BOOST_DIR)/lib" "-Wl,-rpath,$(BOOST_DIR)/lib"
endif

OBJ_FILES = \
	$(TARGET_PATH)/Alchemist.o \
	$(TARGET_PATH)/Executor.o \
	$(TARGET_PATH)/Driver.o \
	$(TARGET_PATH)/Worker.o \
	#

$(TARGET_PATH)/%.o: $(SRC_PATH)/%.cpp $(SRC_PATH)/Parameters.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

.PHONY: default
default: $(TARGET_PATH)/alchemist

$(TARGET_PATH)/alchemist: $(TARGET_PATH) $(OBJ_FILES)
	$(CXX) -dynamic $(CXXFLAGS) -o $@ $(OBJ_FILES) $(LDFLAGS)

$(TARGET_PATH):
	mkdir -p $(TARGET_PATH)

.PHONY: clean
clean:
	rm -rf $(TARGET_PATH)
