#!/usr/bin/env bash

ifeq ($(shell uname), Linux)				# Poor heuristic to check if on Cori 
source ./build/Cori/build.sh
else
source ./build/MacOS/build.sh
endif