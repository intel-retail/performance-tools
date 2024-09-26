#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

FROM ubuntu:22.04 as builder
ENV HOME=/home/nobody
ENV no_proxy=localhost,127.0.0.1
ENV NO_PROXY=localhost,127.0.0.1

# don't ask anything
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update  && \
apt-get install -y --no-install-recommends \ 
  python3 \
  python3-pip \
  python3-venv && \
rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN pip install gunicorn[gthread]
WORKDIR /opt/venv
RUN find . -name "pip*" -exec rm -rf {} \; ;exit 0
RUN find . -name "*normalizer*" -exec rm -rf {} \; ;exit 0
RUN find . -name "activate*" -exec rm -rf {} \; ;exit 0
RUN find . -name "Activate*" -exec rm -rf {} \; ;exit 0
RUN find . -name "python-wheels" -exec rm -rf {} \; ;exit 0
RUN find . -name "easy_install*" -exec rm -rf {} \; ;exit 0
RUN find . -name "setuptools*" -exec rm -rf {} \; ;exit 0
RUN find . -name "__pycache__" -exec rm -rf {} \; ;exit 0

FROM ubuntu:22.04

ENV HOME=/home/nobody
ENV no_proxy=localhost,127.0.0.1
ENV NO_PROXY=localhost,127.0.0.1

# don't ask anything
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /tmp/work

RUN apt-get update && \
  apt-get install -y --no-install-recommends wget gnupg2 ca-certificates && \
  wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
  wget https://github.com/intel/xpumanager/releases/download/V1.2.24/xpumanager_1.2.24_20231120.070911.ddc18e8a.u22.04_amd64.deb && \
  echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy flex' | \
  tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    dmidecode \
    python3 \
    libnl-genl-3-200 \
    libmfx-tools \
    intel-gsc \
    intel-level-zero-gpu \
    level-zero \
    libdrm2 && \
  apt-get remove -y wget gnupg2 ca-certificates && \
  apt-get autoremove -y && \
  rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN ldconfig && dpkg -i --force-all *.deb

WORKDIR /

ENV PATH="/opt/venv/bin:$PATH"

COPY entrypoint.sh entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
