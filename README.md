# MetalNeedle: M1 backend for an education deep learning framework Needle

Final project for the class "Deep Learning Systems Algorithms and Implementation" from CMU, where we try to make needle work with Apple M1 chips.

## Prerequisites

To run the code in this repo (with M1 as NDArray backend), we assume you have a Mac machine with M1 chip on it. There are some setup steps you need to follow before you can run our code, as indicated below.

### Step 1. Install necessary compilation tools

First, you need to download XCode and its SDK. You can install Xcode through your Mac’s app store.
To ensure Metal is installed successfully, run this in command line:
`xcrun metal`

if you run into the following error:

> `xcrun: error: unable to find utility "metal", not a developer tool or in PATH`

run the following command to fix it:

```bash
xcode-select --switch /Applications/Xcode.app/Contents/Developer
```

Then, run the following in the Mac terminal (you can skip these if you already have these installed on your system):

```bash
brew install llvm
brew install cmake
```

### Step 2. Install python dependencies

We recommend using conda for installing all the python packages needed for running unit tests in this codebase. After installing conda on your system, run the following to install the conda environment needed for our code.

``` bash
conda env create --file environment.yaml
```

Run the following to activate the environment

``` bash
conda activate dlsys-needle-m1
```

If you want, you can also use pip to install all the packages listed in `environment.yml` and not use conda.

### Step 3. Download data

Some of the unit tests require CIFAR and PTB data. You can download them by running

``` bash
cd hw4
python3 download_data.py
```

**NOTE: you have to be in hw4 directory when you run `download_data.py`, because `download_data.py` hardcodes the data path.**

## Usage

### Run unit tests from hw3 and hw4, now with M1 as backend

Go to the directory containing Makefile

``` bash
cd hw4
```

<!-- compile C++ metal code
```
/opt/homebrew/opt/llvm/bin/clang++ \
    -std=c++17 -stdlib=libc++ -O2 \
    -L/opt/homebrew/opt/libomp/lib -fopenmp \
    -I../../metal-cpp \
    -fno-objc-arc \
    -framework Metal -framework Foundation -framework MetalKit \
    -g main.cpp MetalOperations.cpp  -o main.x
```
Compile M1 GPU kernel code
```
xcrun -sdk macosx metal -c ops.metal -o MyLibrary.air
xcrun -sdk macosx metallib MyLibrary.air -o ops.metallib
``` -->

Compile C++ metal code && M1 GPU kernel code

``` bash
make
```

Then you can run unit tests (Since currently we add `./python` to `sys.path` in pytest, you need to execute the command under project root e.g hw4, but you can execute commands in other dir by adding additional paths)

We combined all local tests from hw3 and hw4, and make it available for m1-backend, except for mugrade tests and language model training(it fails due to our hw4 implementation error)

According to [this PyTorch GitHub issue](https://github.com/pytorch/pytorch/issues/77799), currently sequential models are not friendly for m1 GPUs even for Apple's MPS backend and PyTorch(it takes much longer than CPU), so we reduced seq_len, input_size, hidden_states for rnn and lstm tests so that they can be passed within acceptable time.

Run unit tests on everything:

``` bash
python3 -m pytest -v
```

Run part of the tests according to test names, e.g. "m1":

``` bash
python3 -m pytest -v -k "m1"
```

### benchmark matrix multiplication on CPU vs M1

``` bash
cd hw4
python3 benchmark_matmul.py
```

You should see the following plots that compare the matrix multiplication speed on m1 vs cpu. As we can see, for matrices with size bigger than 100, m1 consistently have ~3x speedup comparing to cpu. In some cases such as when matrix size is 2500, m1 achieves 70x speedup comparing to cpu! (this is likely due to cache misses in the CPU for loop)

![matmul duration comparison](matmul_duration_comparison.png)
![matmul speedup](matmul_speedup.png)
