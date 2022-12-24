# MetalNeedle: M1 backend for an education deep learning framework Needle

Final project for the class "Deep Learning Systems Algorithms and Implementation" from CMU, where we try to make needle work with Apple M1 chips.

# Prerequisites

To run the code in this repo (with M1 as NDArray backend), we assume you have a Mac machine with M1 chip on it. There are some setup steps you need to follow before you can run our code, as indicated below.

## Step 1. Install necessary compilation tools

First, you need to download XCode and SDK. You can install Xcode through your Mac’s app store.
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

## Step 2. Install python dependencies

I used conda for installing all the python packages needed for running unit tests in this codebase. After installing conda on your system, run the following to install the conda environment needed for our code.

```
conda env create --file environment.yaml
```

Run the following to activate the environment

```
conda activate dlsys-needle-m1
```

If you want, you can also use pip to install all the packages listed in `environment.yml` and not use conda.

## Step 3. Download data

Some of the unit tests require CIFAR and PTB data. You can download them by running

```
cd hw4
python3 download_data.py
```

**NOTE: you have to be in hw4 directory when you run `download_data.py`, because `download_data.py` hardcodes the data path.**

# Usage

Go to the directory containing Makefile

```
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

```
make
```

Run unit tests (Since currently we add `./python` to `sys.path` in pytest, you need to execute the command under project root e.g hw4, but you can add other paths as well)

Due to some reason that we don't understand, `test_rnn` and `test_lstm` must be run
individually to complete. If we run them together with everything else, the test will get
stuck.

```
python3 -m pytest -v -k "not test_lstm and not test_rnn and not test_language_model_training"
```

Run the following to test lstm and rnn individually.

```
python3 -m pytest -v -k "test_lstm"
python3 -m pytest -v -k "test_rnn"
```

Run the following to test language model training on M1 (for some reason it fails on the CPU, but succeeds on M1. Might be due to my hw4 implementation error.):

```
python3 -m pytest -v -k "test_language_model_training and m1"
```

<!-- Run the binary
```
cd build
./main.x
```
The `main.cpp` program is pretty simple, as it constructs 3 buffers of length 3 for `a, b, c` (2 input buffers, 1 output buffer), and pass them into M1 GPU kernel for elemenwise addition, then print out results. The expected result is:
```
Running on Apple M1

before add arrays (M1 GPU):
a_CPP:1,1,1,
b_CPP:2,3,4,
c_CPP:0,0,0,
after add arrays (M1 GPU):
a_CPP:1,1,1,
b_CPP:2,3,4,
c_CPP:3,4,5,
``` -->
