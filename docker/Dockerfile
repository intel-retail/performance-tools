FROM ubuntu:25.04@sha256:95a416ad2446813278ec13b7efdeb551190c94e12028707dd7525632d3cec0d1

ARG HTTP_PROXY
ENV PCM_DIRECTORY=/opt/intel

# Optional: Set APT proxy if needed
RUN if [ -n "$HTTP_PROXY" ]; then \
    echo "Acquire::http::Proxy \"$HTTP_PROXY\";" > /etc/apt/apt.conf; \
    fi

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    git \
    iotop \
    sysstat \
    jq \
    curl \
    cmake \
    python3-pip \
    build-essential \
    docker.io \
    pciutils \
    intel-gpu-tools \
    supervisor && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --require-hashes --no-deps --no-cache-dir --break-system-packages -r requirements.txt

# Clean up any old PCM
RUN if [ -d "/opt/intel/pcm" ]; then rm -rf /opt/intel/pcm; fi

# Clone and build PCM
RUN echo "Installing PCM" && \
    [ ! -d "$PCM_DIRECTORY" ] && mkdir -p "$PCM_DIRECTORY" && \
    cd $PCM_DIRECTORY && \
    git clone --recursive https://github.com/opcm/pcm.git && \
    cd $PCM_DIRECTORY/pcm && \
    mkdir build && cd build && \
    cmake .. && cmake --build . && \
    mkdir -p /opt/intel/pcm-bin/bin /opt/intel/pcm-bin/lib && \
    cp -r bin /opt/intel/pcm-bin/ && \
    cp -r lib /opt/intel/pcm-bin/ && \
    rm -rf "$PCM_DIRECTORY/pcm"
    

# Copy all scripts and supervisor config
COPY scripts/ /scripts/
COPY supervisord.conf supervisord.conf

# Ensure all .sh are executable
RUN chmod +x /scripts/*.sh

CMD ["/usr/bin/supervisord", "-c", "/supervisord.conf"]
