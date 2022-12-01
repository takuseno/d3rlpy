from typing import Any, Dict

from .decorators import pretty_repr


@pretty_repr
class Device:
    """GPU Device class.

    This class manages GPU id.

    Args:
        idx: GPU id.

    """

    _idx: int

    def __init__(self, idx: int = 0):
        self._idx = idx

    def get_id(self) -> int:
        """Returns GPU id.

        Returns:
            GPU id.

        """
        return self._idx

    def __eq__(self, obj: Any) -> bool:
        if isinstance(obj, Device):
            return self._idx == obj.get_id()
        raise ValueError("Device cannot be comapred with non Device objects.")

    def __ne__(self, obj: Any) -> bool:
        return not self.__eq__(obj)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"idx": self._idx}
