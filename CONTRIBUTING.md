# Contributing to d3rlpy

Any kind of contribution to d3rlpy would be highly appreciated!

Contribution examples:
- Thumbing up to good issues or pull requests :+1:
- Opening issues about questions, bugs, installation problems, feature requests, algorithm requests etc.
- Sending pull requests

## Development Guide

### build from source
```
$ git clone git@github.com:takuseno/d3rlpy
$ cd d3rlpy
$ pip install -e .
```

Before making your nice PR, please run the follwing commands to inspect code qualities.

### testing
```
$ pip install pytest-cov onnxruntime onnx # dependencies used in unit tests
$ ./scripts/test
```

### coding style
This repository is styled with [black](https://github.com/psf/black) formatter.
Also, [isort](https://github.com/PyCQA/isort) is used to format package imports.
[docformatter](https://github.com/PyCQA/docformatter) is additionally used to format docstrings.
```
$ pip install black isort docformatter # formatters
$ ./scripts/format
```

### linter
This repository is fully type-annotated and checked by [mypy](https://github.com/python/mypy).
Also, [pylint](https://github.com/PyCQA/pylint) checks code consistency.
```
$ pip install mypy pylint==2.13.5 # linters
$ ./scripts/lint
```
