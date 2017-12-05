#!/usr/bin/env bash

export ALCHEMIST_ROOT=$HOME/Projects/AlSuite/Alchemist
export ALLIB_ROOT=$HOME/Projects/AlSuite/AlLib
export ALTEST_ROOT=$HOME/Projects/AlSuite/AlTest

export ALCHEMIST_JAR=$ALCHEMIST_ROOT/target/scala-2.11/alchemist-assembly-0.1.jar
export ALCHEMIST_EXE=$ALCHEMIST_ROOT/target/alchemist

export ALLIB_JAR=$ALLIB_ROOT/target/scala-2.11/allib-assembly-0.1.jar
export ALLIB_DYLIB=$ALLIB_ROOT/target/allib.dylib

cp target/alchemist ../AlTest/lib

# scala ../AlTest/target/scala-2.11/altest-assembly-0.1.jar

export PYTHONPATH="$ALTEST_ROOT/src/main/python/:$PYTHONPATH"
/usr/bin/env python "$ALTEST_ROOT/src/main/python/run_tests.py" "$@"