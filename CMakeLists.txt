cmake_minimum_required(VERSION 3.25)
project(norch_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NORCH_STRICT_WARNINGS "Enable strict compiler warnings with -Werror" ON)
option(NORCH_SANITIZERS "Enable ASan/UBSan for test builds" OFF)

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
set(NORCH_HAS_NCCL OFF)
set(NORCH_CUDA_INCLUDE_DIR "")
set(NORCH_NCCL_INCLUDE_DIR "")
set(NORCH_CUDART_LIB "")
set(NORCH_NCCL_LIB "")
set(NORCH_EXTRA_RPATHS "")

if(TORCH_CUDA_LIB AND C10_CUDA_LIB)
  file(GLOB NVIDIA_CU_INCLUDE_HINTS "${LIBTORCH_DIR}/../nvidia/cu*/include" "${LIBTORCH_DIR}/../nvidia/cuda_runtime/include")
  find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
    HINTS ${NVIDIA_CU_INCLUDE_HINTS} /usr/local/cuda/include /usr/include
  )
  file(GLOB NVIDIA_CU_LIB_HINTS "${LIBTORCH_DIR}/../nvidia/cu*/lib" "${LIBTORCH_DIR}/../nvidia/cuda_runtime/lib")
  find_library(CUDART_LIB
    NAMES cudart libcudart.so.13 libcudart.so.12
    HINTS ${NVIDIA_CU_LIB_HINTS} /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
  )
  if(CUDA_RUNTIME_INCLUDE_DIR)
    set(NORCH_HAS_CUDA ON)
    set(NORCH_CUDA_INCLUDE_DIR "${CUDA_RUNTIME_INCLUDE_DIR}")
    if(CUDART_LIB)
      set(NORCH_CUDART_LIB "${CUDART_LIB}")
      get_filename_component(CUDART_DIR "${CUDART_LIB}" DIRECTORY)
      list(APPEND NORCH_EXTRA_RPATHS "${CUDART_DIR}")
    endif()
    # Check NCCL headers for c10d::ProcessGroupNCCL
    file(GLOB NVIDIA_NCCL_INC_HINTS "${LIBTORCH_DIR}/../nvidia/nccl/include")
    find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NVIDIA_NCCL_INC_HINTS} /usr/local/cuda/include /usr/include)
    file(GLOB NVIDIA_NCCL_LIB_HINTS "${LIBTORCH_DIR}/../nvidia/nccl/lib")
    find_library(NCCL_LIB NAMES nccl libnccl.so.2 HINTS ${NVIDIA_NCCL_LIB_HINTS} /usr/lib/x86_64-linux-gnu)
    if(NCCL_INCLUDE_DIR)
      set(NORCH_HAS_NCCL ON)
      set(NORCH_NCCL_INCLUDE_DIR "${NCCL_INCLUDE_DIR}")
      if(NCCL_LIB)
        set(NORCH_NCCL_LIB "${NCCL_LIB}")
        get_filename_component(NCCL_DIR "${NCCL_LIB}" DIRECTORY)
        list(APPEND NORCH_EXTRA_RPATHS "${NCCL_DIR}")
      endif()
    endif()
    message(STATUS "norch: CUDA libtorch detected (nccl=${NORCH_HAS_NCCL}, headers=${NORCH_CUDA_INCLUDE_DIR})")
  else()
    message(STATUS "norch: libtorch_cuda found but cuda_runtime_api.h missing; building without CUDA runtime headers")
  endif()
else()
  message(STATUS "norch: CPU-only libtorch detected")
endif()

set(NORCH_SOURCES
  native/src/addon.cpp
  native/src/addon_state.cpp
  native/src/tensor_wrap.cpp
  native/src/generator_wrap.cpp
  native/src/convert.cpp
  native/src/dispatcher.cpp
  native/src/autograd.cpp
  native/src/ops_fast.cpp
  native/src/cuda.cpp
  native/src/distributed.cpp
  native/src/export_runner.cpp
)

add_library(norch_native SHARED ${NORCH_SOURCES})

set(NORCH_RPATH_LIST "${TORCH_LIB_DIR}" ${NORCH_EXTRA_RPATHS})
list(REMOVE_DUPLICATES NORCH_RPATH_LIST)

set_target_properties(norch_native PROPERTIES
  PREFIX ""
  SUFFIX ".node"
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${NORCH_RPATH_LIST}"
  INSTALL_RPATH "${NORCH_RPATH_LIST}"
  BUILD_RPATH_USE_ORIGIN ON
)

target_include_directories(norch_native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/native/include"
)

target_include_directories(norch_native SYSTEM PRIVATE
  "${NODE_INCLUDE_DIR}"
  "${NODE_ADDON_API_DIR}"
  ${TORCH_INCLUDE_DIRS}
  $<$<BOOL:${NORCH_HAS_CUDA}>:${NORCH_CUDA_INCLUDE_DIR}>
  $<$<BOOL:${NORCH_HAS_NCCL}>:${NORCH_NCCL_INCLUDE_DIR}>
)

target_compile_definitions(norch_native PRIVATE
  NAPI_VERSION=9
  NAPI_CPP_EXCEPTIONS
  BUILDING_NODE_EXTENSION
  _GLIBCXX_ASSERTIONS
  $<$<BOOL:${NORCH_HAS_CUDA}>:NORCH_HAS_CUDA=1>
  $<$<BOOL:${NORCH_HAS_NCCL}>:NORCH_HAS_NCCL=1>
  $<$<BOOL:${NORCH_HAS_NCCL}>:USE_C10D_NCCL=1>
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
  set(NORCH_TORCH_LINK_LIBS
    -Wl,--no-as-needed
    "${TORCH_LIB}"
    "${TORCH_CPU_LIB}"
    "${C10_LIB}"
  )
  if(NORCH_HAS_CUDA)
    list(APPEND NORCH_TORCH_LINK_LIBS "${TORCH_CUDA_LIB}" "${C10_CUDA_LIB}")
    if(NORCH_CUDART_LIB)
      list(APPEND NORCH_TORCH_LINK_LIBS "${NORCH_CUDART_LIB}")
    endif()
    if(NORCH_HAS_NCCL AND NORCH_NCCL_LIB)
      list(APPEND NORCH_TORCH_LINK_LIBS "${NORCH_NCCL_LIB}")
    endif()
  endif()
  if(TORCH_GLOBAL_DEPS_LIB)
    list(APPEND NORCH_TORCH_LINK_LIBS "${TORCH_GLOBAL_DEPS_LIB}")
  endif()
  list(APPEND NORCH_TORCH_LINK_LIBS -Wl,--as-needed)
  target_link_libraries(norch_native PRIVATE ${NORCH_TORCH_LINK_LIBS})
endif()

target_precompile_headers(norch_native PRIVATE
  <ATen/ATen.h>
  <napi.h>
)
