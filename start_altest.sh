#!/usr/bin/env bash


if [ "$HOSTNAME" = "cori04" ]; then			# Not a fool-proof heuristic to check if on Cori 
source ./build/Cori/config.sh
else
source ./build/MacOS/config.sh
endif

export ALCHEMIST_JAR=$ALCHEMIST_ROOT/target/scala-2.11/alchemist-assembly-0.1.jar
export ALCHEMIST_EXE=$ALCHEMIST_ROOT/target/alchemist

export ALLIB_JAR=$ALLIB_ROOT/target/scala-2.11/allib-assembly-0.1.jar
export ALLIB_DYLIB=$ALLIB_ROOT/target/allib.dylib

# scala AlTest/target/scala-2.11/altest-assembly-0.1.jar

export CURDIR=$PWD
cd $ALTEST_ROOT

export PYTHONPATH="$ALTEST_ROOT/src/main/python/:$PYTHONPATH"
/usr/bin/env python "$ALTEST_ROOT/src/main/python/run_tests.py" "$@"

cd $CURDIR