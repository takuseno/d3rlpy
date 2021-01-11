FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# this needs to avoid time zone question
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      software-properties-common \
      cmake \
      python3-dev \
      zlib1g \
      zlib1g-dev \
      libgl1-mesa-dri \
      libgl1-mesa-glx \
      libglu1-mesa-dev \
      libasio-dev \
      git \
      pkg-config \
      python3-tk \
      libsm6 \
      libxext6 \
      libxrender1 \
      libpcre3-dev && \
    pip install --no-cache-dir \
      Cython \
      git+https://github.com/takuseno/d4rl-pybullet \
      git+https://github.com/takuseno/d4rl-atari && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    git clone https://github.com/takuseno/d3rlpy && \
    cd d3rlpy && \
    pip install --no-cache-dir . && \
    cd .. && \
    rm -rf d3rlpy

EXPOSE 6006

CMD ["tail", "-f", "/dev/null"]
