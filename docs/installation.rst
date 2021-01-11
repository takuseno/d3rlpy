Installation
============

Recommended Platforms
---------------------

d3rlpy supports Linux, macOS and also Windows.


Install d3rlpy
--------------

Install via PyPI
~~~~~~~~~~~~~~~~

`pip` is a recommended way to install d3rlpy::

  $ pip install d3rlpy

Install via Anaconda
~~~~~~~~~~~~~~~~~~~~

d3rlpy is also available on `conda-forge`::

  $ conda install -c conda-forge d3rlpy


Install via Docker
~~~~~~~~~~~~~~~~~~

d3rlpy is also available on Docker Hub::

  $ docker run -it --gpus all --name d3rlpy takuseno/d3rlpy:latest bash


Install from source
~~~~~~~~~~~~~~~~~~~

You can also install via GitHub repository::

  $ git clone https://github.com/takuseno/d3rlpy
  $ cd d3rlpy
  $ pip install Cython numpy # if you have not installed them.
  $ pip install -e .
