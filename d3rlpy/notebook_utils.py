import base64

__all__ = ["start_virtual_display", "render_video"]


def start_virtual_display() -> None:
    """Starts virtual display."""
    try:
        from pyvirtualdisplay.display import Display

        display = Display()
        display.start()
    except ImportError as e:
        raise ImportError(
            "pyvirtualdisplay is not installed.\n"
            "$ pip install pyvirtualdisplay"
        ) from e


def render_video(path: str) -> None:
    """Renders video file in Jupyter Notebook.

    Args:
        path: Path to video file.
    """
    try:
        from IPython import display as ipythondisplay
        from IPython.core.display import HTML

        with open(path, "r+b") as f:
            encoded = base64.b64encode(f.read())
            template = """
                <video alt="test" autoplay loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>
            """
            ipythondisplay.display(
                HTML(data=template.format(encoded.decode("ascii")))
            )
    except ImportError as e:
        raise ImportError(
            "This should be executed inside Jupyter Notebook."
        ) from e
