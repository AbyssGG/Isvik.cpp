include_guard(GLOBAL)

set(ISVIK_SLINT_VERSION "1.18.1")
set(ISVIK_SLINT_PREFIX
  "${PROJECT_SOURCE_DIR}/out/_deps/slint-${ISVIK_SLINT_VERSION}"
  CACHE PATH "Install prefix for the pinned Slint C++ SDK")
set(SLINT_STYLE "fluent-light" CACHE STRING "Slint widget style" FORCE)

function(isvik_prepare_slint)
  set(_slint_config "${ISVIK_SLINT_PREFIX}/lib/cmake/Slint/SlintConfig.cmake")
  if(NOT EXISTS "${_slint_config}")
    set(_download_dir "${PROJECT_BINARY_DIR}/_downloads")
    file(MAKE_DIRECTORY "${_download_dir}")

    if(WIN32 AND MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(_slint_url
        "https://github.com/slint-ui/slint/releases/download/v${ISVIK_SLINT_VERSION}/Slint-cpp-${ISVIK_SLINT_VERSION}-win64-MSVC-AMD64.exe")
      set(_slint_package "${_download_dir}/Slint-cpp-${ISVIK_SLINT_VERSION}-win64-MSVC-AMD64.exe")
      file(DOWNLOAD "${_slint_url}" "${_slint_package}"
        EXPECTED_HASH "SHA256=eea71789a54982bc1c164d9e1d2c7f00fead1415e9ac396ccb48676f2f66a7a9"
        TLS_VERIFY ON
        SHOW_PROGRESS
        STATUS _slint_download_status)
      list(GET _slint_download_status 0 _slint_download_code)
      if(NOT _slint_download_code EQUAL 0)
        list(GET _slint_download_status 1 _slint_download_message)
        message(FATAL_ERROR "Slint SDK download failed: ${_slint_download_message}")
      endif()
      execute_process(
        COMMAND "${_slint_package}" /S "/D=${ISVIK_SLINT_PREFIX}"
        RESULT_VARIABLE _slint_install_result)
      if(NOT _slint_install_result STREQUAL "0")
        message(FATAL_ERROR "Slint SDK installer failed: ${_slint_install_result}")
      endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux"
        AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
      set(_slint_url
        "https://github.com/slint-ui/slint/releases/download/v${ISVIK_SLINT_VERSION}/Slint-cpp-${ISVIK_SLINT_VERSION}-Linux-x86_64.tar.gz")
      set(_slint_package "${_download_dir}/Slint-cpp-${ISVIK_SLINT_VERSION}-Linux-x86_64.tar.gz")
      file(DOWNLOAD "${_slint_url}" "${_slint_package}"
        EXPECTED_HASH "SHA256=f0b24ea601b80afe241677d1179bf5215dceae529f2b275b678a0af9a144b255"
        TLS_VERIFY ON
        SHOW_PROGRESS
        STATUS _slint_download_status)
      list(GET _slint_download_status 0 _slint_download_code)
      if(NOT _slint_download_code EQUAL 0)
        list(GET _slint_download_status 1 _slint_download_message)
        message(FATAL_ERROR "Slint SDK download failed: ${_slint_download_message}")
      endif()
      file(MAKE_DIRECTORY "${ISVIK_SLINT_PREFIX}")
      file(ARCHIVE_EXTRACT INPUT "${_slint_package}" DESTINATION "${ISVIK_SLINT_PREFIX}")
    else()
      message(FATAL_ERROR
        "No pinned Slint ${ISVIK_SLINT_VERSION} binary package is available for this platform. "
        "Set ISVIK_SLINT_PREFIX to a compatible Slint C++ SDK install prefix.")
    endif()
  endif()

  find_package(Slint ${ISVIK_SLINT_VERSION} EXACT CONFIG REQUIRED
    PATHS "${ISVIK_SLINT_PREFIX}/lib/cmake/Slint"
    NO_DEFAULT_PATH)
  if(NOT TARGET Slint::Slint)
    message(FATAL_ERROR "The Slint package did not provide the Slint::Slint target")
  endif()
endfunction()
