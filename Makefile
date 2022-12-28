.PHONY: lib, pybind, clean, format, all

METAL_SRC=ndarray_backend/ops.metal

all: lib ops.metallib 

ops.metallib: $(METAL_SRC)
	@mkdir -p build
	xcrun -sdk macosx metal -c $(METAL_SRC) -o ./build/MyLibrary.air
	xcrun -sdk macosx metallib ./build/MyLibrary.air -o ./build/$@

lib:
	@mkdir -p build
	@cd build; CXX=$(CXX) cmake ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i ndarray_backend/*.cc ndarray_backend/*.cu ndarray_backend/*.cpp ndarray_backend/*.hpp

clean:
	rm -rf build needle/backend_ndarray/ndarray_backend*.so
