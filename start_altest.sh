#!/usr/bin/env bash

export ALCHEMIST_ROOT=$HOME/Projects/AlSuite/Alchemist
export ALLIB_ROOT=$HOME/Projects/AlSuite/AlLib
export ALTEST_ROOT=$HOME/Projects/AlSuite/AlTest

export ALCHEMIST_JAR=$ALCHEMIST_ROOT/target/scala-2.11/alchemist-assembly-0.1.jar
export ALCHEMIST_EXE=$ALCHEMIST_ROOT/target/alchemist

export ALLIB_JAR=$ALLIB_ROOT/target/scala-2.11/allib-assembly-0.1.jar
export ALLIB_DYLIB=$ALLIB_ROOT/target/allib.dylib

scala AlTest/target/scala-2.11/altest-assembly-0.1.jar