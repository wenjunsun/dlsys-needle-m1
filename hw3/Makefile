.PHONY: lib, pybind, clean, format, all

CXX=/opt/homebrew/opt/llvm/bin/clang++
CXX_FLAG=-std=c++17 -stdlib=libc++ -O2 -fno-objc-arc
CXX_FLAG+=-I../../metal-cpp
CXX_FLAG+=-framework Metal -framework Foundation -framework MetalKit

CXX_SRC=src/main.cpp src/MetalOperations.cpp
METAL_SRC=src/ops.metal

all: main.x ops.metallib lib

main.x: $(CXX_SRC)
	@mkdir -p build
	@cd src; $(CXX) $(CXX_FLAG) -o ../build/$@ $(notdir $^)

ops.metallib:
	@mkdir -p build
	@cd build; xcrun -sdk macosx metal -c ../src/ops.metal -o MyLibrary.air
	@cd build; xcrun -sdk macosx metallib MyLibrary.air -o $@

lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so main.x.dSYM
