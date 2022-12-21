# dlsys-needle-m1
Final project for the class "Deep Learning Systems Algorithms and Implementation" from CMU, where we try to make needle work with Apple M1 chips.

# directions to compile M1 code and run the trial program in `hw3/src/main.cpp`
Go to the directory containing M1 code
```
cd hw3/src
```
compile C++ metal code
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
```
run the binary
```
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
```
