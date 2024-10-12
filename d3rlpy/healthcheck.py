__all__ = ["run_healthcheck"]


def run_healthcheck() -> None:
    _check_gym()


def _check_gym() -> None:
    import gymnasium
    from gym.version import VERSION

    if VERSION < "0.26.0":
        raise ValueError(
            "Gym version is too outdated. "
            "Please upgrade Gym to 0.26.0 or later."
        )

    if gymnasium.__version__ < "1.0.0":
        raise ValueError(
            "Gymnasium version is too outdated. "
            "Please upgrade Gymnasium to 1.0.0 or later."
        )
