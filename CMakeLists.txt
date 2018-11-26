cmake_minimum_required(VERSION 3.9)
project(mpilab3)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

option(INTEL OFF)

if (INTEL)
#    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O2 -qopt-report=2 -mavx2 -unroll -use-intel-optimized-headers -qopt-prefetch=5  -qopt-prefetch-issue-excl-hint -qopt-prefetch-distance=60000,8000")
    add_definitions(-DINTEL)
    add_definitions(-DL2CACHE_SIZE=256000)
    add_definitions(-DOPTIMAL_THREADS=4)
endif()
#set(CMAKE_CXX_FLAGS "-qopt-report=5 -simd -msse4.2")

add_executable(mpilab3 main.cpp)
add_executable(tests tests.cpp)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(mpilab3 PUBLIC ${MPI_CXX_LIBRARIES})

message(${MPI_INCLUDE_PATH})
message(${MPI_CXX_LIBRARIES})

find_package(Threads)
target_link_libraries(mpilab3 PUBLIC ${CMAKE_THREAD_LIBS_INIT})
