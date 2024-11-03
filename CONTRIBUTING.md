# Contributing to d3rlpy

Any kind of contribution to d3rlpy would be highly appreciated!

Contribution examples:
- Thumbing up to good issues or pull requests :+1:
- Opening issues about questions, bugs, installation problems, feature requests, algorithm requests etc.
- Sending pull requests

## Development Guide

### Build from source
```
$ git clone git@github.com:takuseno/d3rlpy
$ cd d3rlpy
$ pip install -e .
```

Before making your nice PR, please run the follwing commands to inspect code qualities.

### Install additional dependencies for development
```
$ pip install -r dev.requirements.txt
```

### Testing
```
$ ./scripts/test
```

### Coding style check
This repository is styled and analyzed with [Ruff](https://docs.astral.sh/ruff/).
[docformatter](https://github.com/PyCQA/docformatter) is additionally used to format docstrings.
This repository is fully type-annotated and checked by [mypy](https://github.com/python/mypy).
Before you submit your PR, please execute this command:
```
$ ./scripts/lint
```
