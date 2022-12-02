# DSAA 6000D Project



## Dependencies and prerequisites
1. CUDA >= 10.0
2. GCC = 7.5.0
3. Boost = 1.77.0
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
```