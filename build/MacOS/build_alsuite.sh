#!/usr/bin/env bash

source ./build/MacOS/config.sh

export ALCHEMIST_JAR=$ALCHEMIST_ROOT/target/scala-2.11/alchemist-assembly-0.1.jar
export ALCHEMIST_EXE=$ALCHEMIST_ROOT/target/alchemist

export ALLIB_JAR=$ALLIB_ROOT/target/scala-2.11/allib-assembly-0.1.jar
export ALLIB_DYLIB=$ALLIB_ROOT/target/allib.dylib

echo " "
echo "Building AlSuite for MacOS"
echo "=========================="
echo " "
echo "Building Alchemist"
echo "------------------"
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
echo "Building AlLib"
echo "--------------"
echo " "
echo "Entered 'AlLib/'"
cd $ALLIB_ROOT
mkdir -p lib/

cp ../Alchemist/target/scala-2.11/alchemist-assembly-0.1.jar ./lib
cp ../Alchemist/target/alchemist ./lib

echo "Creating AlLib jar:"
echo " "
sbt -batch assembly
echo " "
echo "Creating AlLib dylib:"
echo " "
make
echo " "
echo "Done with AlLib"

echo " "
echo "Building AlTest"
echo "---------------"
echo " "
echo "Entered 'AlTest/'"
cd $ALTEST_ROOT
mkdir -p lib/

cp ../Alchemist/target/scala-2.11/alchemist-assembly-0.1.jar ./lib
cp ../Alchemist/target/alchemist ./lib

cp ../AlLib/target/scala-2.11/allib-assembly-0.1.jar ./lib
cp ../AlLib/target/allib.dylib ./lib

echo "Creating AlTest jar:"
echo " "
sbt -batch assembly

echo " "
echo "Building process for AlSuite has completed"
echo "If no issues occurred during building, configure 'AlTest/config/config.py' and then run 'start_altest.sh'"
