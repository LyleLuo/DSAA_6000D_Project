# DSAA 6000D Project



## Dependencies and prerequisites
1. CUDA >= 10.0 & <= 11.4
2. GCC >= 7.5.0
3. Boost with iostreams and serialization
4. Python >= 3.7

## Build
```sh
PROJ_HOME='The directory of the project'
pip install pybind11

cd $PROJ_HOME
git submodule update --init

cd $PROJ_HOME/extern/pybind11
mkdir build
cd build
cmake ..
make check -j 4

cd $PROJ_HOME/ads_int
make adds
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Run
`read_gr.py is an example`
```
import adds

# interface 1
adds.sssp_from_file(input_file_name, output_file_name)

# interface 2
result = adds.sssp_from_csr(num_nodes, num_edges, indptr_array, indices_array, graph_csr_array)
```

## Dataset

https://zenodo.org/record/4365954/files/sssp-int.zip?download=1
