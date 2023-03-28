# Dockerfile for asnets
#
# To build:
# docker build -t asnets-bionic -f asnets-bionic.dockerfile .
#
# To run:
# docker run -i --rm --mount type=bind,source=<full_path_to_shared_dir>,target=/home/asnets_user/shared -t asnets-bionic /bin/bash

# Base container.
FROM ubuntu:bionic

# Install packages.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       python3-numpy python3-dev python3-pip python3-wheel python3-venv flex \
       bison build-essential autoconf libtool git libboost-all-dev cmake \
       libhdf5-dev g++ git make

# Set up asnets user and group.
RUN groupadd -g 999 asnets_user \
    && useradd -r -u 999 -g asnets_user asnets_user \
    && mkdir /home/asnets_user \
    && chown -R asnets_user:asnets_user /home/asnets_user \
    && adduser asnets_user sudo \
    && echo "asnets ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER asnets_user

# Make sure LD_LIBRARY_PATH is set correctly to include /usr/local/lib.
ENV LD_LIBRARY_PATH /usr/local/lib:/usr/local/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}

WORKDIR /home/asnets_user

# Install ASNets
RUN git clone https://github.com/qxcv/asnets.git

RUN python3 -m venv venv-asnets && . venv-asnets/bin/activate \
    && pip3 install --upgrade pip \
    && pip3 install wheel cython numpy pkgconfig protobuf==3.19.6 werkzeug

RUN . venv-asnets/bin/activate && pip3 install -e asnets/asnets

RUN echo 'source ~/venv-asnets/bin/activate' >> ~/.bashrc

RUN echo 'ulimit -S -v 64000000' >> ~/.bashrc
