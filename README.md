# QLA Practical Examples
Practical examples from the paper **"From Block Encodings to Generalized Quantum Signal Processing: Principles, Algorithms and Applications"**.

The code included in this repository corresponds to the following examples:

### 3.3 Low-Rank Approximation of Images and Spectral Filtering
This example illustrates how to implement a spectral filter to produce a low-rank approximation of an image using QSVT.

The relevant files are:
  - [spectral_filtering.ipynb](spectral_filtering.ipynb)
  - [data/airplane.jpg](data/airplane.jpg)
  - [QSP.py](QSP.py)

### 3.4 Matrix Logarithm
Given an $\epsilon_0$-accurate implementation of $U_H = e^{iH}$ for Hermitian matrix $H$ with $\|H\|<\frac{\pi}{4}$, this example illustrates how to use the matrix logarithm technique to allow the implementation of a block-encoding of $H$.

The relevant files are: 
  - [matrix_logarithm.ipynb](matrix_logarithm.ipynb)
  - [QSP.py](QSP.py)
  - [QSVT_circuit.py](QSVT_circuit.py)

### 3.6 Quantum Linear Systems
This example illustrates how to implement matrix inversion in order to solve a linear system efficiently, using QSVT.

The relevant files are:
  - [linear_systems.ipynb](linear_systems.ipynb)
  - [QSP.py](QSP.py)

### 3.7 Computational Finance
This example illustrates how to solve the Black-Scholes equation under certain conditions.

The relevant file is:
  - [computational_finance.ipynb](computational_finance.ipynb)
