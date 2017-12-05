#!/usr/bin/env bash

export ALCHEMIST_ROOT=$HOME/Projects/AlSuite/Alchemist
export ALLIB_ROOT=$HOME/Projects/AlSuite/AlLib
export ALTEST_ROOT=$HOME/Projects/AlSuite/AlTest

export ALCHEMIST_JAR=$ALCHEMIST_ROOT/target/scala-2.11/alchemist-assembly-0.1.jar
export ALCHEMIST_EXE=$ALCHEMIST_ROOT/target/alchemist

export ALLIB_JAR=$ALLIB_ROOT/target/scala-2.11/allib-assembly-0.1.jar
export ALLIB_DYLIB=$ALLIB_ROOT/target/allib.dylib

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
echo "Build of Alchemist Suite completed"
echo "Run 'start_altest.sh'"
