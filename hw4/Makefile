.PHONY: lib, pybind, clean, format, all

METAL_SRC=src/ops.metal

all: lib ops.metallib 

ops.metallib: $(METAL_SRC)
	xcrun -sdk macosx metal -c $(METAL_SRC) -o python/needle/backend_ndarray/MyLibrary.air
	xcrun -sdk macosx metallib python/needle/backend_ndarray/MyLibrary.air -o python/needle/backend_ndarray/$@

lib:
	@mkdir -p build
	@cd build; CXX=$(CXX) cmake ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so python/needle/backend_ndarray/ops.metallib python/needle/backend_ndarray/MyLibrary.air
