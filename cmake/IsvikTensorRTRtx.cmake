include_guard(GLOBAL)

set(ISVIK_TENSORRT_RTX_EP_VERSION "0.4.2")
set(ISVIK_TENSORRT_RTX_EP_SHA256
  "1dbc8f927c63bb9d3b885f51f1e3786f78308a2de2a1965d42bbba03383d3b6a")
set(ISVIK_TENSORRT_RTX_EP_PREFIX
  "${PROJECT_SOURCE_DIR}/out/_deps/tensorrt-rtx-ep-abi-${ISVIK_TENSORRT_RTX_EP_VERSION}-cu13"
  CACHE PATH "Install prefix for the pinned NVIDIA TensorRT-RTX EP ABI runtime")

function(isvik_prepare_tensorrt_rtx_ep)
  if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "The pinned TensorRT-RTX EP runtime is currently configured for Windows x64 MSVC.")
  endif()

  set(_provider "${ISVIK_TENSORRT_RTX_EP_PREFIX}/onnxruntime_providers_nv_tensorrt_rtx.dll")
  if(NOT EXISTS "${_provider}")
    set(_download_dir "${PROJECT_SOURCE_DIR}/out/_downloads")
    file(MAKE_DIRECTORY "${_download_dir}")
    set(_archive "${_download_dir}/TensorRT-RTX-EP-ABI-v${ISVIK_TENSORRT_RTX_EP_VERSION}-cu13.zip")
    file(DOWNLOAD
      "https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases/download/v${ISVIK_TENSORRT_RTX_EP_VERSION}/TensorRT-RTX-EP-ABI-v${ISVIK_TENSORRT_RTX_EP_VERSION}-cu13.zip"
      "${_archive}"
      EXPECTED_HASH "SHA256=${ISVIK_TENSORRT_RTX_EP_SHA256}"
      TLS_VERIFY ON
      SHOW_PROGRESS
      STATUS _download_status)
    list(GET _download_status 0 _download_code)
    if(NOT _download_code EQUAL 0)
      list(GET _download_status 1 _download_message)
      message(FATAL_ERROR "TensorRT-RTX EP runtime download failed: ${_download_message}")
    endif()
    file(MAKE_DIRECTORY "${ISVIK_TENSORRT_RTX_EP_PREFIX}")
    file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${ISVIK_TENSORRT_RTX_EP_PREFIX}")
  endif()

  if(NOT EXISTS "${_provider}" OR NOT EXISTS "${ISVIK_TENSORRT_RTX_EP_PREFIX}/tensorrt_rtx_1_6.dll")
    message(FATAL_ERROR "The TensorRT-RTX EP archive is missing its provider or runtime DLL")
  endif()
  set(ISVIK_TENSORRT_RTX_EP_ROOT "${ISVIK_TENSORRT_RTX_EP_PREFIX}" PARENT_SCOPE)
endfunction()
