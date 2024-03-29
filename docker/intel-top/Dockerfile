#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#


FROM ubuntu:20.04
RUN if [ -n "$HTTP_PROXY" ] ; then  echo "Acquire::http::Proxy \"$HTTP_PROXY\";" >  /etc/apt/apt.conf; fi
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    cmake  \
    libunwind-dev \
    libgsl-dev \
    libasound2-dev \
    libxmlrpc-core-c3-dev \
    libjson-c-dev \
    libcurl4-openssl-dev \
    python-docutils \
    valgrind \
    peg \
    libdrm-intel1 \
    pkg-config libdrm-dev libkmod-dev libprocps-dev libdw-dev libpixman-1-dev libcairo-dev libudev-dev flex bison \
    meson && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install igt
WORKDIR /igt

# Install 1.27.1
RUN git config --global http.proxy $HTTP_PROXY; git clone https://gitlab.freedesktop.org/drm/igt-gpu-tools.git; cd igt-gpu-tools; git checkout 2b29e8ac07fbcfadc48b9d60e4d736a6e3b289ab

RUN cd  igt-gpu-tools; meson build; ninja -C build; cd build; ninja install
COPY entrypoint.sh entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]