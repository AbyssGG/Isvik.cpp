include_guard(GLOBAL)

set(ISVIK_OPENVINO_GENAI_VERSION "2026.4.0.0")
set(ISVIK_OPENVINO_GENAI_SHA256
  "478a05c5ea30da26db3ae637477d553303cf38f070641fb4f218daf6d4c0a2e9")
set(ISVIK_OPENVINO_GENAI_PREFIX
  "${PROJECT_SOURCE_DIR}/out/_deps/openvino-genai-${ISVIK_OPENVINO_GENAI_VERSION}"
  CACHE PATH "Install prefix for the pinned OpenVINO GenAI C++ SDK")
set(ISVIK_OPENVINO_GENAI_ROOT
  "${ISVIK_OPENVINO_GENAI_PREFIX}/openvino_genai_windows_${ISVIK_OPENVINO_GENAI_VERSION}_x86_64")

function(isvik_prepare_openvino_genai)
  set(_runtime_cmake "${ISVIK_OPENVINO_GENAI_ROOT}/runtime/cmake")
  set(_genai_config "${_runtime_cmake}/OpenVINOGenAIConfig.cmake")
  if(NOT EXISTS "${_genai_config}")
    if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
      message(FATAL_ERROR
        "The pinned OpenVINO GenAI package is currently provided for Windows x64 MSVC only. "
        "Set ISVIK_ENABLE_OPENVINO_GENAI=OFF on other platforms.")
    endif()

    set(_download_dir "${PROJECT_SOURCE_DIR}/out/_downloads")
    set(_package "${_download_dir}/openvino_genai_windows_${ISVIK_OPENVINO_GENAI_VERSION}_x86_64.zip")
    file(MAKE_DIRECTORY "${_download_dir}")
    file(DOWNLOAD
      "https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/${ISVIK_OPENVINO_GENAI_VERSION}/windows/openvino_genai_windows_${ISVIK_OPENVINO_GENAI_VERSION}_x86_64.zip"
      "${_package}"
      EXPECTED_HASH "SHA256=${ISVIK_OPENVINO_GENAI_SHA256}"
      TLS_VERIFY ON
      SHOW_PROGRESS
      STATUS _download_status)
    list(GET _download_status 0 _download_code)
    if(NOT _download_code EQUAL 0)
      list(GET _download_status 1 _download_message)
      message(FATAL_ERROR "OpenVINO GenAI SDK download failed: ${_download_message}")
    endif()

    file(MAKE_DIRECTORY "${ISVIK_OPENVINO_GENAI_PREFIX}")
    file(ARCHIVE_EXTRACT INPUT "${_package}" DESTINATION "${ISVIK_OPENVINO_GENAI_PREFIX}")
    if(NOT EXISTS "${_genai_config}")
      message(FATAL_ERROR "OpenVINO GenAI SDK archive did not contain ${_genai_config}")
    endif()
  endif()

  # GenAI is built against the Runtime bundled in the same archive. Keep both
  # packages on that exact ABI instead of accidentally finding another install.
  set(OpenVINO_DIR "${_runtime_cmake}" CACHE PATH "OpenVINO Runtime bundled with GenAI" FORCE)
  set(OpenVINOGenAI_DIR "${_runtime_cmake}" CACHE PATH "Pinned OpenVINO GenAI package" FORCE)
  set(ISVIK_OPENVINO_GENAI_ROOT "${ISVIK_OPENVINO_GENAI_ROOT}" PARENT_SCOPE)
endfunction()
