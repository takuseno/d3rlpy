import os

from setuptools import setup, Extension

os.environ['CFLAGS'] = '-std=c++11'

# get __version__ variable
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'd3rlpy', '_version.py')).read())

if __name__ == "__main__":
    from numpy import get_include
    from Cython.Build import cythonize

    # setup Cython build
    ext = Extension('d3rlpy.dataset',
                    sources=['d3rlpy/dataset.pyx'],
                    include_dirs=[get_include(), 'd3rlpy/cpp/include'],
                    language='c++',
                    extra_compile_args=["-std=c++11", "-O3", "-ffast-math"],
                    extra_link_args=["-std=c++11"])

    ext_modules = cythonize([ext],
                            compiler_directives={
                                'linetrace': True,
                                'binding': True
                            })

    # main setup
    setup(name="d3rlpy",
          version=__version__,
          description="Data-driven Deep Reinforcement Learning Library as an Out-of-the-box Tool",
          long_description=open("README.md").read(),
          long_description_content_type="text/markdown",
          url="https://github.com/takuseno/d3rlpy",
          author="Takuma Seno",
          author_email="takuma.seno@gmail.com",
          license="MIT License",
          classifiers=["Development Status :: 4 - Beta",
                       "Intended Audience :: Developers",
                       "Intended Audience :: Education",
                       "Intended Audience :: Science/Research",
                       "Topic :: Scientific/Engineering",
                       "Topic :: Scientific/Engineering :: Artificial Intelligence",
                       "Programming Language :: Python :: 3.6",
                       "Programming Language :: Python :: 3.7",
                       "Programming Language :: Python :: 3.8",
                       "Programming Language :: Python :: Implementation :: CPython",
                       "Operating System :: POSIX :: Linux",
                       'Operating System :: Microsoft :: Windows',
                       "Operating System :: MacOS :: MacOS X"],
          install_requires=["torch",
                            "scikit-learn",
                            "tensorboardX",
                            "tqdm",
                            "GPUtil",
                            "h5py",
                            "gym",
                            "kornia",
                            "click",
                            "typing-extensions",
                            "cloudpickle",
                            "scipy",
                            "structlog"],
          packages=["d3rlpy",
                    "d3rlpy.algos",
                    "d3rlpy.algos.torch",
                    "d3rlpy.augmentation",
                    "d3rlpy.dynamics",
                    "d3rlpy.dynamics.torch",
                    "d3rlpy.envs",
                    "d3rlpy.iterators",
                    "d3rlpy.metrics",
                    "d3rlpy.models",
                    "d3rlpy.models.torch",
                    "d3rlpy.online",
                    "d3rlpy.ope",
                    "d3rlpy.ope.torch",
                    "d3rlpy.preprocessing",
                    "d3rlpy.wrappers"],
          python_requires=">=3.6.0",
          zip_safe=False,
          package_data={'d3rlpy': ['*.pyx',
                                   '*.pxd',
                                   '*.h',
                                   '*.pyi',
                                   'py.typed']},
          ext_modules=ext_modules,
          entry_points={'console_scripts': ['d3rlpy=d3rlpy.cli:cli']})
