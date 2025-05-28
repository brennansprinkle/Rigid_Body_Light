cmake_minimum_required(VERSION 3.22)
project(c_rigid_obj)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
set(CMAKE_BUILD_TYPE Release)
add_compile_definitions(PUBLIC MAXLOGLEVEL=3)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)


find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG)
find_package (Eigen3 REQUIRED NO_MODULE)
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})
# include_directories(${CMAKE_SOURCE_DIR}/include)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif()


# Ensure ${Python3_SOABI} is not empty
if(NOT Python3_SOABI)
  message(FATAL_ERROR "Python3_SOABI is empty. Set it to the output of 'python3-config --extension-suffix'")
endif()

add_library(c_rigid_obj.${Python3_SOABI} SHARED c_rigid_obj.cpp)
set_target_properties(c_rigid_obj.${Python3_SOABI} PROPERTIES SUFFIX ".so") # force .so suffix for mac
set_target_properties(c_rigid_obj.${Python3_SOABI} PROPERTIES PREFIX "") # remove the lib prefix

target_link_libraries(c_rigid_obj.${Python3_SOABI} PRIVATE pybind11::pybind11 Eigen3::Eigen)
set_target_properties(c_rigid_obj.${Python3_SOABI} PROPERTIES PREFIX "")
target_compile_options(c_rigid_obj.${Python3_SOABI} PRIVATE -fPIC -Wall -O3)

install(TARGETS c_rigid_obj.${Python3_SOABI} DESTINATION ${Python3_SITEARCH})