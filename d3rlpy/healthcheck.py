import warnings

__all__ = ["run_healthcheck"]


def run_healthcheck() -> None:
    _check_gym()


def _check_gym() -> None:
    from gym.version import VERSION

    if VERSION < "0.26.2":
        warnings.warn(
            f"gym=={VERSION} is outdated. "
            "Please upgrade Gym to 0.26.2 or later."
        )
