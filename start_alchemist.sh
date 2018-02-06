#!/usr/bin/env bash

export ALCHEMIST_ROOT=$HOME/Projects/AlSuite/Alchemist

export ALCHEMIST_JAR=$ALCHEMIST_ROOT/target/scala-2.11/alchemist-assembly-0.1.jar
export ALCHEMIST_EXE=$ALCHEMIST_ROOT/target/alchemist

# spark-submit --master local[3] --class alchemist.Alchemist Alchemist/target/scala-2.11/alchemist-assembly-0.1.jar

spark-shell --jars Alchemist/target/scala-2.11/alchemist-assembly-0.1.jar
