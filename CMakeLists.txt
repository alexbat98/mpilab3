cmake_minimum_required(VERSION 3.12)
project(mpilab3)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(mpilab3 main.cpp)
add_executable(tests tests.cpp)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(mpilab3 PUBLIC ${MPI_CXX_LIBRARIES})