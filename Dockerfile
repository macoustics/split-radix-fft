# Use Debian Bookworm as the base image
FROM debian:bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    clang \
    cmake \
    git \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace
CMD ["/bin/bash"]
