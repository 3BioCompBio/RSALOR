# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.10)
project(computeWeightsBackend)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Source files
set(SRC_FILES 
    computeWeightsBackend.cpp 
    msa.cpp 
)

# Shared library output
add_library(computeWeightsBackend SHARED ${SRC_FILES})

# Change output name to _computeWeightsBackend.so
set_target_properties(computeWeightsBackend PROPERTIES
    OUTPUT_NAME "_computeWeightsBackend"
    SUFFIX ".so"
)

# Compiler options (optional: add any optimization/debug flags here)
target_compile_options(computeWeightsBackend PRIVATE -fPIC -Ofast)

# Link libraries (add more if needed, e.g., -lpthread)
target_link_libraries(computeWeightsBackend PRIVATE)