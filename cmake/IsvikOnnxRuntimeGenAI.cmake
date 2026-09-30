include_guard(GLOBAL)

set(ISVIK_ONNXRUNTIME_GENAI_VERSION "0.17.0")
set(ISVIK_ONNXRUNTIME_GENAI_SHA256
  "0bc5c187f9d6bd91582b7a0439fd47474a2a3cbc67941e6144eb11e0d09bb0b3")
set(ISVIK_ONNXRUNTIME_GENAI_CUDA_SHA256
  "df59044e4c6c6556bf2c532c62c4a95e0043f442d76495b5e81f5df5d65e54ec")
set(ISVIK_ONNXRUNTIME_VERSION "1.26.0")
set(ISVIK_ONNXRUNTIME_SHA256
  "6ebe99b5564bf4d029b6e93eac9ff423682b6212eade769e9ca3f685eaf500b4")

set(ISVIK_ONNXRUNTIME_GENAI_PREFIX
  "${PROJECT_SOURCE_DIR}/out/_deps/onnxruntime-genai-${ISVIK_ONNXRUNTIME_GENAI_VERSION}"
  CACHE PATH "Install prefix for the pinned ONNX Runtime GenAI C++ SDK")
set(ISVIK_ONNXRUNTIME_GENAI_ROOT
  "${ISVIK_ONNXRUNTIME_GENAI_PREFIX}/onnxruntime-genai-${ISVIK_ONNXRUNTIME_GENAI_VERSION}-win-x64")
set(ISVIK_ONNXRUNTIME_GENAI_CUDA_PREFIX
  "${PROJECT_SOURCE_DIR}/out/_deps/onnxruntime-genai-cuda-${ISVIK_ONNXRUNTIME_GENAI_VERSION}"
  CACHE PATH "Install prefix for the pinned ONNX Runtime GenAI CUDA device add-on")
set(ISVIK_ONNXRUNTIME_GENAI_CUDA_ROOT
  "${ISVIK_ONNXRUNTIME_GENAI_CUDA_PREFIX}/onnxruntime-genai-${ISVIK_ONNXRUNTIME_GENAI_VERSION}-win-x64-cuda")
set(ISVIK_ONNXRUNTIME_PREFIX
  "${PROJECT_SOURCE_DIR}/out/_deps/onnxruntime-${ISVIK_ONNXRUNTIME_VERSION}"
  CACHE PATH "Install prefix for the pinned ONNX Runtime C++ SDK")
set(ISVIK_ONNXRUNTIME_ROOT
  "${ISVIK_ONNXRUNTIME_PREFIX}/onnxruntime-win-x64-${ISVIK_ONNXRUNTIME_VERSION}")

function(isvik_prepare_onnxruntime_genai)
  if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR
      "The pinned ONNX Runtime GenAI packages are currently configured for Windows x64 MSVC. "
      "Set ISVIK_ENABLE_ONNXRUNTIME_GENAI=OFF on other platforms.")
  endif()

  set(_download_dir "${PROJECT_SOURCE_DIR}/out/_downloads")
  file(MAKE_DIRECTORY "${_download_dir}")

  set(_genai_config "${ISVIK_ONNXRUNTIME_GENAI_ROOT}/lib/cmake/onnxruntime-genai/onnxruntime-genaiConfig.cmake")
  if(NOT EXISTS "${_genai_config}")
    set(_genai_archive "${_download_dir}/onnxruntime-genai-${ISVIK_ONNXRUNTIME_GENAI_VERSION}-win-x64.zip")
    file(DOWNLOAD
      "https://github.com/microsoft/onnxruntime-genai/releases/download/v${ISVIK_ONNXRUNTIME_GENAI_VERSION}/onnxruntime-genai-${ISVIK_ONNXRUNTIME_GENAI_VERSION}-win-x64.zip"
      "${_genai_archive}"
      EXPECTED_HASH "SHA256=${ISVIK_ONNXRUNTIME_GENAI_SHA256}"
      TLS_VERIFY ON
      SHOW_PROGRESS
      STATUS _genai_download_status)
    list(GET _genai_download_status 0 _genai_download_code)
    if(NOT _genai_download_code EQUAL 0)
      list(GET _genai_download_status 1 _genai_download_message)
      message(FATAL_ERROR "ONNX Runtime GenAI SDK download failed: ${_genai_download_message}")
    endif()
    file(MAKE_DIRECTORY "${ISVIK_ONNXRUNTIME_GENAI_PREFIX}")
    file(ARCHIVE_EXTRACT INPUT "${_genai_archive}" DESTINATION "${ISVIK_ONNXRUNTIME_GENAI_PREFIX}")
  endif()

  if(NOT EXISTS "${ISVIK_ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
    set(_ort_archive "${_download_dir}/onnxruntime-win-x64-${ISVIK_ONNXRUNTIME_VERSION}.zip")
    file(DOWNLOAD
      "https://github.com/microsoft/onnxruntime/releases/download/v${ISVIK_ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ISVIK_ONNXRUNTIME_VERSION}.zip"
      "${_ort_archive}"
      EXPECTED_HASH "SHA256=${ISVIK_ONNXRUNTIME_SHA256}"
      TLS_VERIFY ON
      SHOW_PROGRESS
      STATUS _ort_download_status)
    list(GET _ort_download_status 0 _ort_download_code)
    if(NOT _ort_download_code EQUAL 0)
      list(GET _ort_download_status 1 _ort_download_message)
      message(FATAL_ERROR "ONNX Runtime SDK download failed: ${_ort_download_message}")
    endif()
    file(MAKE_DIRECTORY "${ISVIK_ONNXRUNTIME_PREFIX}")
    file(ARCHIVE_EXTRACT INPUT "${_ort_archive}" DESTINATION "${ISVIK_ONNXRUNTIME_PREFIX}")
  endif()

  set(ISVIK_ONNXRUNTIME_GENAI_ROOT "${ISVIK_ONNXRUNTIME_GENAI_ROOT}" PARENT_SCOPE)
  set(ISVIK_ONNXRUNTIME_ROOT "${ISVIK_ONNXRUNTIME_ROOT}" PARENT_SCOPE)
endfunction()

function(isvik_prepare_onnxruntime_genai_cuda)
  if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR
      "The pinned ONNX Runtime GenAI CUDA add-on is currently configured for Windows x64 MSVC.")
  endif()

  set(_download_dir "${PROJECT_SOURCE_DIR}/out/_downloads")
  file(MAKE_DIRECTORY "${_download_dir}")
  set(_cuda_addon "${ISVIK_ONNXRUNTIME_GENAI_CUDA_ROOT}/lib/onnxruntime-genai-cuda.dll")
  if(NOT EXISTS "${_cuda_addon}")
    set(_cuda_archive "${_download_dir}/onnxruntime-genai-${ISVIK_ONNXRUNTIME_GENAI_VERSION}-win-x64-cuda.zip")
    file(DOWNLOAD
      "https://github.com/microsoft/onnxruntime-genai/releases/download/v${ISVIK_ONNXRUNTIME_GENAI_VERSION}/onnxruntime-genai-${ISVIK_ONNXRUNTIME_GENAI_VERSION}-win-x64-cuda.zip"
      "${_cuda_archive}"
      EXPECTED_HASH "SHA256=${ISVIK_ONNXRUNTIME_GENAI_CUDA_SHA256}"
      TLS_VERIFY ON
      SHOW_PROGRESS
      STATUS _cuda_download_status)
    list(GET _cuda_download_status 0 _cuda_download_code)
    if(NOT _cuda_download_code EQUAL 0)
      list(GET _cuda_download_status 1 _cuda_download_message)
      message(FATAL_ERROR "ONNX Runtime GenAI CUDA add-on download failed: ${_cuda_download_message}")
    endif()
    file(MAKE_DIRECTORY "${ISVIK_ONNXRUNTIME_GENAI_CUDA_PREFIX}")
    file(ARCHIVE_EXTRACT INPUT "${_cuda_archive}" DESTINATION "${ISVIK_ONNXRUNTIME_GENAI_CUDA_PREFIX}")
  endif()

  if(NOT EXISTS "${_cuda_addon}")
    message(FATAL_ERROR "ONNX Runtime GenAI CUDA package did not contain ${_cuda_addon}")
  endif()
  set(ISVIK_ONNXRUNTIME_GENAI_CUDA_ROOT "${ISVIK_ONNXRUNTIME_GENAI_CUDA_ROOT}" PARENT_SCOPE)
endfunction()
