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
This repository is styled with [black](https://github.com/psf/black) formatter.
Also, [isort](https://github.com/PyCQA/isort) is used to format package imports.
[docformatter](https://github.com/PyCQA/docformatter) is additionally used to format docstrings.
```
$ ./scripts/format
```

### Linter
This repository is fully type-annotated and checked by [mypy](https://github.com/python/mypy).
Also, [pylint](https://github.com/PyCQA/pylint) checks code consistency.
```
$ ./scripts/lint
```
