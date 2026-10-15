cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
endif()
if("${USE_CUDA}" STREQUAL "OFF")
  message(FATAL_ERROR "MPK requires CUDA")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()
project(MIRAGE_MPK LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_subdirectory(deps/json)
file(GLOB_RECURSE MIRAGE_MPK_SRCS CONFIGURE_DEPENDS
  src/base/*.cc src/kernel/*.cc src/kernel/*.cu
  src/threadblock/*.cc src/threadblock/*.cu src/utils/*.cc)
list(APPEND MIRAGE_MPK_SRCS src/layout.cc)
add_library(mirage_runtime STATIC ${MIRAGE_MPK_SRCS})
set_target_properties(mirage_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(mirage_runtime PUBLIC cxx_std_17 cuda_std_17)
target_compile_definitions(mirage_runtime PUBLIC
  MIRAGE_BACKEND_USE_CUDA MIRAGE_FINGERPRINT_USE_CUDA)
target_include_directories(mirage_runtime PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/deps/cutlass/include
  ${PROJECT_SOURCE_DIR}/deps/cutlass/tools/util/include)
target_link_libraries(mirage_runtime PUBLIC
  CUDA::cudart CUDA::cuda_driver CUDA::cublas nlohmann_json::nlohmann_json
  OpenMP::OpenMP_CXX)
install(TARGETS mirage_runtime ARCHIVE DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
