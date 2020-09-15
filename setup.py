from setuptools import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

# setup Cython build
ext = Extension('d3rlpy.dataset',
                sources=['d3rlpy/dataset.pyx'],
                include_dirs=[get_include()])

setup(name="d3rlpy",
      version="0.23",
      description="Data-driven Deep Reinforcement Learning Library as an Out-of-the-box Tool",
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      url="https://github.com/takuseno/d3rlpy",
      author="Takuma Seno",
      author_email="takuma.seno@gmail.com",
      license="MIT License",
      install_requires=["torch",
                        "scikit-learn",
                        "tensorboardX",
                        "tqdm",
                        "GPUtil",
                        "h5py",
                        "gym",
                        "kornia",
                        "Cython"],
      packages=["d3rlpy",
                "d3rlpy.algos",
                "d3rlpy.algos.torch",
                "d3rlpy.augmentation",
                "d3rlpy.dynamics",
                "d3rlpy.dynamics.torch",
                "d3rlpy.metrics",
                "d3rlpy.models",
                "d3rlpy.models.torch",
                "d3rlpy.preprocessing",
                "d3rlpy.online"],
      python_requires=">=3.5.0",
      zip_safe=False,
      package_data={'d3rlpy': ['*.pyx']},
      ext_modules=cythonize([ext], compiler_directives={'linetrace': True}))
