# pylint: disable=redefined-builtin

import os
import json
import glob
from typing import List, TYPE_CHECKING, Optional

import numpy as np
import click

from . import algos
from ._version import __version__


if TYPE_CHECKING:
    import matplotlib.pyplot


def print_stats(path: str) -> None:
    data = np.loadtxt(path, delimiter=",")
    print("FILE NAME  : ", path)
    print("EPOCH      : ", data[-1, 0])
    print("TOTAL STEPS: ", data[-1, 1])
    print("MAX VALUE  : ", np.max(data[:, 2]))
    print("MIN VALUE  : ", np.min(data[:, 2]))
    print("STD VALUE  : ", np.std(data[:, 2]))


def get_plt() -> "matplotlib.pyplot":
    import matplotlib.pyplot as plt

    try:
        # enable seaborn style if avaiable
        import seaborn as sns

        sns.set()
    except ImportError:
        pass
    return plt


@click.group()
def cli() -> None:
    print("d3rlpy command line interface (Version %s)" % __version__)


@cli.command(short_help="Show statistics of save metrics.")
@click.argument("path")
def stats(path: str) -> None:
    print_stats(path)


@cli.command(short_help="Plot saved metrics (requires matplotlib).")
@click.argument("path", nargs=-1)
@click.option(
    "--window", default=1, show_default=True, help="moving average window."
)
@click.option("--show-steps", is_flag=True, help="use iterations on x-axis.")
@click.option("--show-max", is_flag=True, help="show maximum value.")
def plot(
    path: List[str], window: int, show_steps: bool, show_max: bool
) -> None:
    plt = get_plt()

    max_y_values = []
    min_x_values = []
    max_x_values = []

    for p in path:
        data = np.loadtxt(p, delimiter=",")

        # moving average
        y_data = np.convolve(data[:, 2], np.ones(window) / window, mode="same")

        # create label
        if len(p.split(os.sep)) > 1:
            label = "/".join(p.split(os.sep)[-2:])
        else:
            label = p

        if show_steps:
            x_data = data[:, 1]
        else:
            x_data = data[:, 0]

        max_y_values.append(np.max(data[:, 2]))
        min_x_values.append(np.min(x_data))
        max_x_values.append(np.max(x_data))

        # show statistics
        print("")
        print_stats(p)

        plt.plot(x_data, y_data, label=label)

    if show_max:
        plt.plot(
            [np.min(min_x_values), np.max(max_x_values)],
            [np.max(max_y_values), np.max(max_y_values)],
            color="black",
            linestyle="dashed",
        )

    plt.xlabel("steps" if show_steps else "epochs")
    plt.ylabel("value")
    plt.legend()
    plt.show()


@cli.command(short_help="Plot saved metrics in a grid (requires matplotlib).")
@click.argument("path")
def plot_all(path: str) -> None:
    plt = get_plt()

    # print params.json
    if os.path.exists(os.path.join(path, "params.json")):
        with open(os.path.join(path, "params.json"), "r") as f:
            params = json.loads(f.read())
        print("")
        for k, v in params.items():
            print("%s=%s" % (k, v))

    metrics_names = sorted(list(glob.glob(os.path.join(path, "*.csv"))))
    n_cols = int(np.ceil(len(metrics_names) ** 0.5))
    n_rows = int(np.ceil(len(metrics_names) / n_cols))

    plt.figure(figsize=(12, 7))

    for i in range(n_rows):
        for j in range(n_cols):
            index = j + n_cols * i
            if index >= len(metrics_names):
                break

            plt.subplot(n_rows, n_cols, index + 1)

            data = np.loadtxt(metrics_names[index], delimiter=",")

            plt.plot(data[:, 0], data[:, 2])
            plt.title(os.path.basename(metrics_names[index]))
            plt.xlabel("epoch")
            plt.ylabel("value")

    plt.tight_layout()
    plt.show()


@cli.command(short_help="Export saved model as inference model format.")
@click.argument("path")
@click.option(
    "--format",
    default="onnx",
    show_default=True,
    help="model format (torchscript, onnx).",
)
@click.option(
    "--params-json", default=None, help="explicitly specify params.json."
)
@click.option("--out", default=None, help="output path.")
def export(
    path: str, format: str, params_json: Optional[str], out: Optional[str]
) -> None:
    # check format
    if format not in ["onnx", "torchscript"]:
        raise ValueError("Please specify onnx or torchscript.")

    # find params.json
    if params_json is None:
        dirname = os.path.dirname(path)
        if not os.path.exists(os.path.join(dirname, "params.json")):
            raise RuntimeError(
                "params.json is not found in %s. Please specify"
                "the path to params.json by --params-json."
            )
        params_json = os.path.join(dirname, "params.json")

    # load params
    with open(params_json, "r") as f:
        params = json.loads(f.read())

    # load saved model
    print("Loading %s..." % path)
    algo = getattr(algos, params["algorithm"]).from_json(params_json)
    algo.load_model(path)

    if out is None:
        ext = "onnx" if format == "onnx" else "torchscript"
        export_name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(os.path.dirname(path), export_name + "." + ext)

    # export inference model
    print("Exporting to %s..." % out)
    algo.save_policy(out, as_onnx=format == "onnx")
