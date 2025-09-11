cmake_minimum_required(VERSION 3.24)

project(rapids_singlecell_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Helper to declare a nanobind CUDA module uniformly
function(add_nb_cuda_module target src)
  nanobind_add_module(${target} STABLE_ABI LTO
      ${src}
  )
  target_link_libraries(${target} PRIVATE CUDA::cudart)
  set_target_properties(${target} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
  )
  install(TARGETS ${target} LIBRARY DESTINATION rapids_singlecell/_cuda)
  # Also copy built module into source tree for editable installs
  add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:${target}>
          ${PROJECT_SOURCE_DIR}/src/rapids_singlecell/_cuda/$<TARGET_FILE_NAME:${target}>
  )
endfunction()

# CUDA modules
add_nb_cuda_module(_mean_var_cuda     src/rapids_singlecell/_cuda/mean_var/mean_var.cu)
add_nb_cuda_module(_sparse2dense_cuda src/rapids_singlecell/_cuda/sparse2dense/sparse2dense.cu)
add_nb_cuda_module(_scale_cuda        src/rapids_singlecell/_cuda/scale/scale.cu)
add_nb_cuda_module(_qc_cuda           src/rapids_singlecell/_cuda/qc/qc.cu)
add_nb_cuda_module(_qc_dask_cuda      src/rapids_singlecell/_cuda/qc_dask/qc_kernels_dask.cu)
