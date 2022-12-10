from distutils.core import setup

with open("README.md", "r") as f:
  long_description = f.read()

setup(name="gradflow",
      version="0.0.2",
      description="Simple and lightweight neural network library with a pytorch-like API.",
      author="Javier Sancho",
      license="MIT",
      long_description=long_description,
      packages=["gradflow"],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
      ],
      install_requires=["numpy"],
      python_requires=">=3.8",
      )
