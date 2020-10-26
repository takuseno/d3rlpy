Installation
============

Recommended Platforms
---------------------

d3rlpy only supports Linux and macOS.


Install d3rlpy
--------------

Install via PyPI
~~~~~~~~~~~~~~~

`pip` is a recommended way to install d3rlpy.
Before installing d3rlpy, please make sure if Cython and numpy have been
installed on your system::

  $ pip install Cython numpy

And then, install ``d3rlpy``::

  $ pip install d3rlpy

Install from source
~~~~~~~~~~~~~~~~~~~

You can also install via GitHub repository::

  $ git clone https://github.com/takuseno/d3rlpy
  $ cd d3rlpy
  $ pip install -r requirements.txt
  $ pip install -e .
