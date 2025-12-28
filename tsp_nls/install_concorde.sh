#!/bin/bash
mkdir concorde
cd concorde
mkdir qsopt
cd qsopt
# Download qsopt
if [[ "$OSTYPE" == "darwin"* ]]; then
    curl -O http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt.a
    curl -O http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt.h
    curl -O http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt
else
    wget https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.a
    wget https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.h
    wget https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt
fi
cd ..
wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar xf co031219.tgz
cd concorde
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Detect macOS architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        # Apple Silicon - Concorde doesn't support ARM64 macOS
        echo "Concorde TSP solver does not support Apple Silicon (ARM64) macOS."
        echo "This is expected - Concorde is a legacy codebase from 2003."
        echo "TSP functionality will work without Concorde (local search only)."
        echo "For exact TSP solving, consider using alternative solvers."
        exit 1
    elif [[ "$ARCH" == "x86_64" ]]; then
        # Intel macOS - try configure without host flag first (autodetect)
        if ! ./configure --with-qsopt=$(pwd)/../qsopt; then
            echo "Autodetect failed, trying explicit x86_64 configuration..."
            ./configure --with-qsopt=$(pwd)/../qsopt --host=x86_64-apple-darwin
        fi
    else
        # Unknown architecture - try autodetect
        ./configure --with-qsopt=$(pwd)/../qsopt
    fi
else
    ./configure --with-qsopt=$(realpath ../qsopt)
fi
make
TSP/concorde -s 99 -k 100
cd ../..