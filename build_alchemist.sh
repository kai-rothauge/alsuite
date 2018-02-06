#!/usr/bin/env bash

if [ "$HOSTNAME" = "cori04" ]; then				# Not a fool-proof heuristic to check if on Cori 
source ./build/Cori/build_alchemist.sh
else
source ./build/MacOS/build_alchemist.sh
fi