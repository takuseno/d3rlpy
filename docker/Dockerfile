FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime


RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# d3rlpy
RUN apt-get install -y gcc libxml2-dev libxslt1-dev zlib1g-dev g++ git cmake
RUN pip install Cython
RUN pip install d3rlpy
RUN pip install git+https://github.com/takuseno/d4rl-pybullet
RUN pip install git+https://github.com/takuseno/d4rl-atari

# pip
WORKDIR /tmp
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# clearn
RUN rm -rf /var/lib/apt/lists/*
