cmake_minimum_required(VERSION 3.22)
project(c_rigid_obj)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
set(CMAKE_BUILD_TYPE Release)
add_compile_definitions(PUBLIC MAXLOGLEVEL=3)

enable_language(CXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

set(SOURCES src/c_rigid_obj.cpp)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package (Eigen3 REQUIRED NO_MODULE)
include_directories(${Python_INCLUDE_DIRS})
# include_directories(${CMAKE_SOURCE_DIR}/include)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif()

if(DOUBLEPRECISION)
  add_compile_definitions(PUBLIC DOUBLE_PRECISION)
endif()

# # Ensure ${Python_SOABI} is not empty
# if(NOT Python_SOABI)
#   message(FATAL_ERROR "Python_SOABI is empty. Set it to the output of 'python3-config --extension-suffix'")
# endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
message(STATUS "python executable: ${Python_EXECUTABLE}")
message(STATUS "nanobind_DIR: ${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(c_rigid ${SOURCES})
target_link_libraries(c_rigid PRIVATE Eigen3::Eigen)
target_compile_options(c_rigid PRIVATE -fPIC -Wall -O3)
install(TARGETS c_rigid DESTINATION ${Python_SITEARCH}/Rigid)

# add_library(c_rigid_obj.${Python_SOABI} SHARED ${SOURCES})
# set_target_properties(c_rigid_obj.${Python_SOABI} PROPERTIES SUFFIX ".so") # force .so suffix for mac
# set_target_properties(c_rigid_obj.${Python_SOABI} PROPERTIES PREFIX "") # remove the lib prefix



# target_link_libraries(c_rigid_obj.${Python_SOABI} PRIVATE Eigen3::Eigen)
# set_target_properties(c_rigid_obj.${Python_SOABI} PROPERTIES PREFIX "")
# target_compile_options(c_rigid_obj.${Python_SOABI} PRIVATE -fPIC -Wall -O3)

# install(TARGETS c_rigid_obj.${Python_SOABI} DESTINATION ${Python3_SITEARCH}/Rigid)
file(GLOB PYTHON_SOURCES src/*.py)
install(FILES ${PYTHON_SOURCES} DESTINATION ${Python_SITEARCH}/Rigid)