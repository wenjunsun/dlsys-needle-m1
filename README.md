# MetalNeedle: M1 backend for an education deep learning framework Needle
Final project for the class "Deep Learning Systems Algorithms and Implementation" from CMU, where we try to make needle work with Apple M1 chips.

# Usage
Go to the directory containing Makefile
```
cd hw3
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
Run unit tests for M1 (Since currently the kernel library path is hardcoded, you need to execute the command under the same directory with the kernel library).
```
python3 -m pytest -k "m1"
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
