cmake_minimum_required(VERSION 3.22)
project(c_rigid_obj LANGUAGES CXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
add_compile_definitions(PUBLIC MAXLOGLEVEL=3)
if(DOUBLEPRECISION)
  add_compile_definitions(PUBLIC DOUBLE_PRECISION)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package (Eigen3 REQUIRED NO_MODULE)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
find_package(nanobind CONFIG REQUIRED)

set(SOURCES src/c_rigid_obj.cpp)
nanobind_add_module(c_rigid ${SOURCES})
target_link_libraries(c_rigid PRIVATE Eigen3::Eigen)
target_compile_options(c_rigid PRIVATE -fPIC -Wall)
install(TARGETS c_rigid DESTINATION ${Python_SITEARCH}/Rigid)

file(GLOB PYTHON_SOURCES src/*.py)
install(FILES ${PYTHON_SOURCES} DESTINATION ${Python_SITEARCH}/Rigid)