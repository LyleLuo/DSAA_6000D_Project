
all: sssp  

sssp: kernel.cu support.cu csr_graph.cu
	nvcc -ccbin=g++ $^ -O3 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75  -rdc=true -lcudadevrt -Xptxas -O3  -Xptxas -v -lcuda -lcudart -o $@

adds: py_ads.cu support.cu csr_graph.cu
	nvcc -ccbin=c++ -O3 -shared --compiler-options -fPIC -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 $(shell python3-config --includes) -I../extern/pybind11/include $^ -rdc=true -lcudadevrt -Xptxas -v -lcuda -lcudart -o $@$(shell python3-config --extension-suffix)

clean_adds:
	rm *.so

clean:
	rm -f sssp *.o


