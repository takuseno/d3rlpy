FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# this needs to avoid time zone question
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      software-properties-common \
      cmake \
      git \
      wget \
      unzip \
      python3-dev \
      zlib1g \
      zlib1g-dev \
      libgl1-mesa-dri \
      libgl1-mesa-glx \
      libglu1-mesa-dev \
      libasio-dev \
      pkg-config \
      python3-tk \
      libsm6 \
      libxext6 \
      libxrender1 \
      libpcre3-dev && \
    pip install --no-cache-dir \
      Cython \
      git+https://github.com/takuseno/d4rl-atari && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    wget https://github.com/takuseno/d3rlpy/archive/master.zip && \
    unzip master.zip && \
    cd d3rlpy-master && \
    pip install --no-cache-dir . && \
    cd .. && \
    rm -rf d3rlpy-master master.zip

EXPOSE 6006

CMD ["tail", "-f", "/dev/null"]
