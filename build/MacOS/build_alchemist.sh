#!/usr/bin/env bash

source ./build/MacOS/config.sh

export ALCHEMIST_JAR=$ALCHEMIST_ROOT/target/scala-2.11/alchemist-assembly-0.1.jar
export ALCHEMIST_EXE=$ALCHEMIST_ROOT/target/alchemist

echo " "
echo "Building Alchemist for MacOS"
echo "============================"
echo " "
echo "Entered 'Alchemist/'"
cd $ALCHEMIST_ROOT
echo "Creating Alchemist jar:"
echo " "
sbt -batch assembly
echo " "
echo "Creating Alchemist executable:"
echo " "
make
cd ..
echo " "
echo "Exited 'Alchemist/'"
echo " "
echo "Building process for Alchemist has completed"
echo "If no issues occurred during building, run 'start_alchemist.sh'"
