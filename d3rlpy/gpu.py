import dataclasses
from typing import Any

__all__ = ["Device"]


@dataclasses.dataclass(frozen=True)
class Device:
    """GPU Device class.

    This class manages GPU id.

    Args:
        idx: GPU id.

    """

    idx: int = 0

    def get_id(self) -> int:
        """Returns GPU id.

        Returns:
            GPU id.

        """
        return self.idx

    def __eq__(self, obj: Any) -> bool:
        if isinstance(obj, Device):
            return self.idx == obj.get_id()
        raise ValueError("Device cannot be comapred with non Device objects.")

    def __ne__(self, obj: Any) -> bool:
        return not self.__eq__(obj)
