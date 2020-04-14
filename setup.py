import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optimiusprimal",
    version="0.0.1",
    author="Luke Pratley",
    author_email="luke.pratley@gmail.com",
    description="Convex Optimization Primal Dual Solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Luke-Pratley/optimusprimal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU GPL v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
          'numpy',
          'PyWavelets'
      ]
)

