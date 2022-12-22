.PHONY: lib, pybind, clean, format, all

METAL_SRC=src/ops.metal

all: lib ops.metallib 

ops.metallib: $(METAL_SRC)
	xcrun -sdk macosx metal -c $(METAL_SRC) -o MyLibrary.air
	xcrun -sdk macosx metallib MyLibrary.air -o $@

lib:
	@mkdir -p build
	@cd build; CXX=$(CXX) cmake ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so ops.metallib MyLibrary.air
