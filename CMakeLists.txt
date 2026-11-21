cmake_minimum_required(VERSION 3.25)
project(norch_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NORCH_STRICT_WARNINGS "Enable strict compiler warnings with -Werror" ON)
option(NORCH_SANITIZERS "Enable ASan/UBSan for test builds" OFF)

# Required inputs passed by scripts/build-native.mjs
if(NOT DEFINED LIBTORCH_DIR)
  if(DEFINED ENV{LIBTORCH})
    set(LIBTORCH_DIR "$ENV{LIBTORCH}")
  else()
    message(FATAL_ERROR "LIBTORCH_DIR must be specified (or set LIBTORCH env var)")
  endif()
endif()

if(NOT DEFINED NODE_INCLUDE_DIR)
  message(FATAL_ERROR "NODE_INCLUDE_DIR must be specified")
endif()

if(NOT DEFINED NODE_ADDON_API_DIR)
  message(FATAL_ERROR "NODE_ADDON_API_DIR must be specified")
endif()

set(TORCH_INCLUDE_DIRS
  "${LIBTORCH_DIR}/include"
  "${LIBTORCH_DIR}/include/torch/csrc/api/include"
)
set(TORCH_LIB_DIR "${LIBTORCH_DIR}/lib")

find_library(TORCH_LIB torch PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_CPU_LIB torch_cpu PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(C10_LIB c10 PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

find_library(TORCH_CUDA_LIB torch_cuda PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH)
find_library(C10_CUDA_LIB c10_cuda PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH)
find_library(TORCH_GLOBAL_DEPS_LIB torch_global_deps PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH)

set(NORCH_HAS_CUDA OFF)
if(TORCH_CUDA_LIB AND C10_CUDA_LIB)
  set(NORCH_HAS_CUDA ON)
  message(STATUS "norch: CUDA libtorch detected (${TORCH_CUDA_LIB})")
else()
  message(STATUS "norch: CPU-only libtorch detected")
endif()

set(NORCH_SOURCES
  native/src/addon.cpp
  native/src/addon_state.cpp
  native/src/tensor_wrap.cpp
  native/src/convert.cpp
  native/src/dispatcher.cpp
  native/src/autograd.cpp
  native/src/ops_fast.cpp
  native/src/cuda.cpp
)

add_library(norch_native SHARED ${NORCH_SOURCES})

set_target_properties(norch_native PROPERTIES
  PREFIX ""
  SUFFIX ".node"
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${TORCH_LIB_DIR}"
  INSTALL_RPATH "${TORCH_LIB_DIR}"
  BUILD_RPATH_USE_ORIGIN ON
)

target_include_directories(norch_native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/native/include"
)

target_include_directories(norch_native SYSTEM PRIVATE
  "${NODE_INCLUDE_DIR}"
  "${NODE_ADDON_API_DIR}"
  ${TORCH_INCLUDE_DIRS}
)

target_compile_definitions(norch_native PRIVATE
  NAPI_VERSION=9
  NAPI_CPP_EXCEPTIONS
  BUILDING_NODE_EXTENSION
  _GLIBCXX_ASSERTIONS
  $<$<BOOL:${NORCH_HAS_CUDA}>:NORCH_HAS_CUDA=1>
)

target_compile_options(norch_native PRIVATE
  -O2
  -g
  -fexceptions
  -frtti
)

if(NORCH_STRICT_WARNINGS)
  target_compile_options(norch_native PRIVATE
    -Wall
    -Wextra
    -Wconversion
    -Wshadow
    -Wimplicit-fallthrough
    -Werror
  )
endif()

if(NORCH_SANITIZERS)
  target_compile_options(norch_native PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(norch_native PRIVATE -fsanitize=address,undefined)
endif()

# Prefer mold, fallback to lld
find_program(MOLD_LINKER mold)
find_program(LLD_LINKER ld.lld)
if(MOLD_LINKER)
  target_link_options(norch_native PRIVATE -fuse-ld=mold)
elseif(LLD_LINKER)
  target_link_options(norch_native PRIVATE -fuse-ld=lld)
endif()

if(APPLE)
  target_link_options(norch_native PRIVATE -undefined dynamic_lookup)
  target_link_libraries(norch_native PRIVATE "${TORCH_LIB}" "${TORCH_CPU_LIB}" "${C10_LIB}")
else()
  # Keep libtorch_cuda and libtorch_global_deps linked with --no-as-needed so
  # static CUDA kernel registrations run when the addon loads.
  set(NORCH_TORCH_LINK_LIBS
    -Wl,--no-as-needed
    "${TORCH_LIB}"
    "${TORCH_CPU_LIB}"
    "${C10_LIB}"
  )
  if(NORCH_HAS_CUDA)
    list(APPEND NORCH_TORCH_LINK_LIBS "${TORCH_CUDA_LIB}" "${C10_CUDA_LIB}")
  endif()
  if(TORCH_GLOBAL_DEPS_LIB)
    list(APPEND NORCH_TORCH_LINK_LIBS "${TORCH_GLOBAL_DEPS_LIB}")
  endif()
  list(APPEND NORCH_TORCH_LINK_LIBS -Wl,--as-needed)
  target_link_libraries(norch_native PRIVATE ${NORCH_TORCH_LINK_LIBS})
endif()

# Precompile heavy ATen/torch headers only for files that include them
target_precompile_headers(norch_native PRIVATE
  <ATen/ATen.h>
  <napi.h>
)
