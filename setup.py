from setuptools import setup, find_packages


setup(name="d3rlpy",
      version="0.1",
      license="MIT",
      description="Data-driven Deep Reinforcement Learning library in scikit-learn style",
      url="https://github.com/takuseno/d3rlpy",
      install_requires=["torch",
                        "scikit-learn",
                        "tensorboardX",
                        "pandas",
                        "tqdm",
                        "Pillow"],
      packages=["d3rlpy",
                "d3rlpy.algos",
                "d3rlpy.algos.torch",
                "d3rlpy.metrics",
                "d3rlpy.models",
                "d3rlpy.models.torch",
                "d3rlpy.preprocessing"])
