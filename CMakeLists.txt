cmake_minimum_required(VERSION 3.24)

project(rapids_singlecell_cuda LANGUAGES CXX)

# Option to disable building compiled extensions (for docs/RTD)
option(RSC_BUILD_EXTENSIONS "Build CUDA/C++ extensions" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (RSC_BUILD_EXTENSIONS)
  enable_language(CUDA)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
  find_package(nanobind CONFIG REQUIRED)
  find_package(CUDAToolkit REQUIRED)

  # Find librmm cmake config.
  # Searches all plausible locations: env vars, Python prefix, base_prefix
  # (survives build isolation), and CI marker files.  Works with conda, pip,
  # pixi, uv, hatch, and cibuildwheel.
  set(_env_roots "")
  # Explicit override
  if(DEFINED ENV{LIBRMM_DIR})
    list(APPEND _env_roots "$ENV{LIBRMM_DIR}/..")
  endif()
  # Environment managers
  foreach(_var CONDA_PREFIX VIRTUAL_ENV PIXI_PROJECT_ROOT)
    if(DEFINED ENV{${_var}})
      list(APPEND _env_roots "$ENV{${_var}}")
    endif()
  endforeach()
  # Python prefix + base_prefix (base_prefix survives build isolation —
  # it points to the outer env even when pip/uv creates a temp venv)
  foreach(_attr prefix base_prefix)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.${_attr})"
      OUTPUT_VARIABLE _pp OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_pp)
      list(APPEND _env_roots "${_pp}")
    endif()
  endforeach()
  # CI/cibuildwheel: CIBW_BEFORE_BUILD writes the librmm path to a marker file
  if(EXISTS "/tmp/.librmm_dir")
    file(READ "/tmp/.librmm_dir" _rmm_marker)
    string(STRIP "${_rmm_marker}" _rmm_marker)
    file(GLOB _marker_hints "${_rmm_marker}/lib*/cmake"
                            "${_rmm_marker}/../rapids_logger/lib*/cmake")
    list(APPEND CMAKE_PREFIX_PATH ${_marker_hints})
  endif()
  # Search each root for rmm + rapids_logger cmake configs
  list(REMOVE_DUPLICATES _env_roots)
  foreach(_root ${_env_roots})
    file(GLOB _hints "${_root}/lib/cmake/rmm"
                     "${_root}/lib/python*/site-packages/librmm/lib*/cmake/rmm"
                     "${_root}/lib/python*/site-packages/rapids_logger/lib*/cmake/rapids_logger")
    foreach(_h ${_hints})
      get_filename_component(_dir "${_h}" DIRECTORY)
      list(APPEND CMAKE_PREFIX_PATH "${_dir}")
    endforeach()
  endforeach()
  message(STATUS "rmm search roots: ${_env_roots}")
  message(STATUS "rmm CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
  find_package(rmm CONFIG REQUIRED)
  message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "RSC_BUILD_EXTENSIONS=OFF -> skipping compiled extensions for docs")
endif()

# Helper to declare a nanobind CUDA module uniformly
function(add_nb_cuda_module target src)
  if (RSC_BUILD_EXTENSIONS)
    nanobind_add_module(${target} STABLE_ABI LTO
        ${src}
    )
    target_link_libraries(${target} PRIVATE CUDA::cudart)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    install(TARGETS ${target} LIBRARY DESTINATION rapids_singlecell/_cuda)
    # Generate type stubs at install time (for wheel installs)
    nanobind_add_stub(${target}_stub
        MODULE ${target}
        OUTPUT rapids_singlecell/_cuda/${target}.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:${target}>
        DEPENDS ${target}
        INSTALL_TIME
        MARKER_FILE rapids_singlecell/_cuda/py.typed
    )
    # Generate type stubs at build time (for editable installs)
    nanobind_add_stub(${target}_stub_dev
        MODULE ${target}
        OUTPUT ${target}.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:${target}>
        DEPENDS ${target}
    )
    # Copy built module + stub into source tree for editable installs
    add_custom_command(TARGET ${target}_stub_dev POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/${target}.pyi
            ${PROJECT_SOURCE_DIR}/src/rapids_singlecell/_cuda/${target}.pyi
        COMMAND ${CMAKE_COMMAND} -E touch
            ${PROJECT_SOURCE_DIR}/src/rapids_singlecell/_cuda/py.typed
    )
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:${target}>
            ${PROJECT_SOURCE_DIR}/src/rapids_singlecell/_cuda/$<TARGET_FILE_NAME:${target}>
    )
  endif()
endfunction()

if (RSC_BUILD_EXTENSIONS)
  # CUDA modules
  add_nb_cuda_module(_mean_var_cuda     src/rapids_singlecell/_cuda/mean_var/mean_var.cu)
  add_nb_cuda_module(_sparse2dense_cuda src/rapids_singlecell/_cuda/sparse2dense/sparse2dense.cu)
  add_nb_cuda_module(_scale_cuda        src/rapids_singlecell/_cuda/scale/scale.cu)
  add_nb_cuda_module(_qc_cuda           src/rapids_singlecell/_cuda/qc/qc.cu)
  add_nb_cuda_module(_qc_dask_cuda      src/rapids_singlecell/_cuda/qc_dask/qc_kernels_dask.cu)
  add_nb_cuda_module(_bbknn_cuda        src/rapids_singlecell/_cuda/bbknn/bbknn.cu)
  add_nb_cuda_module(_norm_cuda         src/rapids_singlecell/_cuda/norm/norm.cu)
  add_nb_cuda_module(_pr_cuda           src/rapids_singlecell/_cuda/pr/pr.cu)
  add_nb_cuda_module(_nn_descent_cuda   src/rapids_singlecell/_cuda/nn_descent/nn_descent.cu)
  add_nb_cuda_module(_aucell_cuda       src/rapids_singlecell/_cuda/aucell/aucell.cu)
  add_nb_cuda_module(_nanmean_cuda      src/rapids_singlecell/_cuda/nanmean/nanmean.cu)
  add_nb_cuda_module(_autocorr_cuda     src/rapids_singlecell/_cuda/autocorr/autocorr.cu)
  add_nb_cuda_module(_cooc_cuda         src/rapids_singlecell/_cuda/cooc/cooc.cu)
  add_nb_cuda_module(_aggr_cuda         src/rapids_singlecell/_cuda/aggr/aggr.cu)
  add_nb_cuda_module(_spca_cuda         src/rapids_singlecell/_cuda/spca/spca.cu)
  add_nb_cuda_module(_ligrec_cuda       src/rapids_singlecell/_cuda/ligrec/ligrec.cu)
  add_nb_cuda_module(_pv_cuda           src/rapids_singlecell/_cuda/pv/pv.cu)
  add_nb_cuda_module(_edistance_cuda    src/rapids_singlecell/_cuda/edistance/edistance.cu)
  add_nb_cuda_module(_hvg_cuda          src/rapids_singlecell/_cuda/hvg/hvg.cu)
  add_nb_cuda_module(_kde_cuda          src/rapids_singlecell/_cuda/kde/kde.cu)
  add_nb_cuda_module(_wilcoxon_ovr_cuda src/rapids_singlecell/_cuda/wilcoxon/wilcoxon_ovr.cu)
  add_nb_cuda_module(_wilcoxon_ovo_cuda src/rapids_singlecell/_cuda/wilcoxon/wilcoxon_ovo.cu)
  # Harmony CUDA modules
  add_nb_cuda_module(_harmony_scatter_cuda   src/rapids_singlecell/_cuda/harmony/scatter/scatter.cu)
  add_nb_cuda_module(_harmony_outer_cuda     src/rapids_singlecell/_cuda/harmony/outer/outer.cu)
  add_nb_cuda_module(_harmony_colsum_cuda    src/rapids_singlecell/_cuda/harmony/colsum/colsum.cu)
  add_nb_cuda_module(_harmony_kmeans_cuda    src/rapids_singlecell/_cuda/harmony/kmeans/kmeans.cu)
  add_nb_cuda_module(_harmony_normalize_cuda src/rapids_singlecell/_cuda/harmony/normalize/normalize.cu)
  add_nb_cuda_module(_harmony_pen_cuda       src/rapids_singlecell/_cuda/harmony/pen/pen.cu)
  add_nb_cuda_module(_harmony_clustering_cuda src/rapids_singlecell/_cuda/harmony/clustering/clustering.cu)
  target_link_libraries(_harmony_clustering_cuda PRIVATE CUDA::cublas)
  add_nb_cuda_module(_harmony_correction_cuda src/rapids_singlecell/_cuda/harmony/correction/correction_fast.cu)
  target_link_libraries(_harmony_correction_cuda PRIVATE CUDA::cublas)
  add_nb_cuda_module(_harmony_correction_batched_cuda src/rapids_singlecell/_cuda/harmony/correction/correction_batched.cu)
  target_link_libraries(_harmony_correction_batched_cuda PRIVATE CUDA::cublas)
  # Wilcoxon binned histogram CUDA module
  add_nb_cuda_module(_wilcoxon_binned_cuda   src/rapids_singlecell/_cuda/wilcoxon_binned/wilcoxon_binned.cu)
  if(rmm_FOUND)
    target_link_libraries(_wilcoxon_ovr_cuda PRIVATE rmm::rmm)
    target_link_libraries(_wilcoxon_ovo_cuda PRIVATE rmm::rmm)
  endif()
endif()
