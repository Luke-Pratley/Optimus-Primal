# Optimus Primal
[![Build Status](https://app.travis-ci.com/Luke-Pratley/Optimus-Primal.svg?branch=master)](https://app.travis-ci.com/Luke-Pratley/Optimus-Primal)
[![codecov](https://codecov.io/gh/Luke-Pratley/Optimus-Primal/branch/master/graph/badge.svg)](https://codecov.io/gh/Luke-Pratley/Optimus-Primal)

A light weight proximal splitting Forward Backward Primal Dual based solver for convex optimization problems. 

The current version supports finding the minimum of f(x) + h(A x) + p(B x) + g(x), where f, h, and p are lower semi continuous and have proximal operators, and g is differentiable. A and B are linear operators.

To learn more about proximal operators and algorithms, visit [proximity operator repository](http://proximity-operator.net/index.html). We suggest that users read the tutorial [G. Chierchia, E. Chouzenoux, P. L. Combettes, and J.-C. Pesquet. "The Proximity Operator Repository. User's guide"](http://proximity-operator.net/download/guide.pdf).

## Install
You can install from the master branch
```
pip install git+https://github.com/Luke-Pratley/Optimus-Primal.git@master#egg=optimusprimal
```
or from a frozen version at [pypi](https://pypi.org/project/optimusprimal/)
```
pip install optimusprimal
```

## Requirements
- Python >= 3.8
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)
- Numpy
- Scipy
### Optional
- Matplotlib (only for examples)

