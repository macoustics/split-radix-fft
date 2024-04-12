#!/bin/bash
set -e

LOCAL=1

# Shell
shell() {
    if [ $LOCAL -eq 1 ]; then
        $@
    else
        docker run \
            -it \
            -v $(pwd):/workspace/ \
            split-radix-fft-dev:latest \
            $@
    fi
}

# Function to perform a standard build
build_normal() {
    echo "Performing a normal build..."
    mkdir -p .build
    shell cmake \
            -B .build \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTS_SPLIT_RADIX_FFT=ON \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    shell cmake --build .build
}

# Function to run tests
run_tests() {
    echo "Running tests..."
    shell ./.build/tests/tests
}

# Function to build Docker image
build_docker() {
    echo "Building Docker image..."
    docker build -t split-radix-fft-dev:latest .
}

# Check if no argument is provided
if [ $# -eq 0 ]; then
    build_normal
    exit 0
fi

# Parse command line options
while getopts ":tds" opt; do
    case ${opt} in
        t )
            build_normal
            run_tests
            ;;
        d )
            build_docker
            ;;
        s )
            shell
            ;;
        \? )
            echo "Usage: cmd [-t] for tests, [-d] for docker build, no option for normal build"
            exit 1
            ;;
    esac
done

