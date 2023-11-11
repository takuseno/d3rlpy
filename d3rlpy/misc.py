__all__ = ["IncrementalCounter"]


class IncrementalCounter:
    _value: int

    def __init__(self, init_value: int = 0):
        self._value = init_value

    def get_value(self) -> int:
        return self._value

    def increment(self) -> int:
        self._value += 1
        return self._value
