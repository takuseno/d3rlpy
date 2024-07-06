import os

from setuptools import find_packages, setup

# get __version__ variable
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, "d3rlpy", "_version.py")).read())

if __name__ == "__main__":
    setup(
        name="d3rlpy",
        version=__version__,
        description="An offline deep reinforcement learning library",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/takuseno/d3rlpy",
        author="Takuma Seno",
        author_email="takuma.seno@gmail.com",
        license="MIT License",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: Implementation :: CPython",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: MacOS :: MacOS X",
        ],
        install_requires=[
            "torch>=2.0.0",
            "tqdm>=4.66.3",
            "h5py",
            "gym>=0.26.0",
            "click",
            "typing-extensions",
            "structlog",
            "colorama",
            "dataclasses-json",
            "gymnasium>=1.0.0a1",
        ],
        packages=find_packages(exclude=["tests*"]),
        python_requires=">=3.8.0",
        zip_safe=True,
        entry_points={"console_scripts": ["d3rlpy=d3rlpy.cli:cli"]},
    )
