cmake_minimum_required(VERSION 3.21)

if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
endif()
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(Isvik.cpp VERSION 0.1.0 LANGUAGES C CXX)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

include(FetchContent)
set(JSON_BuildTests OFF CACHE BOOL "Build nlohmann/json tests" FORCE)
set(JSON_Install OFF CACHE BOOL "Install nlohmann/json targets" FORCE)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa)
FetchContent_MakeAvailable(nlohmann_json)

# SQLite is distributed as a public-domain amalgamation with no package CMake
# project, so build its pinned upstream C source as a small static library.
FetchContent_Declare(
  sqlite3_amalgamation
  URL https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip
  URL_HASH SHA256=1e71ddf93849c6a6ecf58b827c0692073d2dd7ee40196158068f7b29f422e87d)
FetchContent_GetProperties(sqlite3_amalgamation)
if(NOT sqlite3_amalgamation_POPULATED)
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
  FetchContent_Populate(sqlite3_amalgamation)
endif()

FetchContent_Declare(
  cpp_httplib
  URL https://codeload.github.com/yhirose/cpp-httplib/tar.gz/refs/tags/v0.58.0
  URL_HASH SHA256=31932ce8b33f2905472a987dc1984b9d8a7d338083a066021349587cc5f1cfea)
FetchContent_GetProperties(cpp_httplib)
if(NOT cpp_httplib_POPULATED)
  FetchContent_Populate(cpp_httplib)
endif()

set(ISVIK_SQLITE_AMALGAMATION_DIR "${sqlite3_amalgamation_SOURCE_DIR}")
add_library(isvik_sqlite STATIC "${ISVIK_SQLITE_AMALGAMATION_DIR}/sqlite3.c")
target_include_directories(isvik_sqlite PUBLIC "${ISVIK_SQLITE_AMALGAMATION_DIR}")
target_compile_definitions(isvik_sqlite PRIVATE
  SQLITE_THREADSAFE=1
  SQLITE_OMIT_LOAD_EXTENSION
  SQLITE_DQS=0)

option(ISVIK_ENABLE_OPENVINO "Build the OpenVINO runtime integration" ON)
option(ISVIK_ENABLE_TENSORRT "Build the optional NVIDIA TensorRT backend" OFF)
if(WIN32)
  set(_isvik_genai_default ON)
  set(_isvik_onnxruntime_genai_default ON)
  set(_isvik_tensorrt_rtx_ep_default ${ISVIK_ENABLE_TENSORRT})
else()
  set(_isvik_genai_default OFF)
  set(_isvik_onnxruntime_genai_default OFF)
  set(_isvik_tensorrt_rtx_ep_default OFF)
endif()
option(ISVIK_ENABLE_OPENVINO_GENAI "Build OpenVINO GenAI model inference" ${_isvik_genai_default})
option(ISVIK_ENABLE_ONNXRUNTIME_GENAI "Build native ONNX Runtime GenAI model inference" ${_isvik_onnxruntime_genai_default})
option(ISVIK_ENABLE_TENSORRT_RTX_EP "Enable NVIDIA TensorRT-RTX ONNX inference when TensorRT is enabled" ${_isvik_tensorrt_rtx_ep_default})
option(ISVIK_ENABLE_CLANG_TIDY "Run clang-tidy while compiling project targets" OFF)

set(ISVIK_TENSORRT_AVAILABLE OFF)

if(ISVIK_ENABLE_CLANG_TIDY)
  find_program(ISVIK_CLANG_TIDY_EXECUTABLE NAMES clang-tidy REQUIRED)
  set(CMAKE_CXX_CLANG_TIDY "${ISVIK_CLANG_TIDY_EXECUTABLE}")
endif()

include(cmake/IsvikSlint.cmake)
isvik_prepare_slint()

add_subdirectory(src)

add_executable(Isvik lsvik/lsvik.cpp lsvik/cli_app.cpp lsvik/native_dialog.cpp lsvik/api_server.cpp)
target_link_libraries(Isvik PRIVATE isvik_core Slint::Slint)
target_include_directories(Isvik SYSTEM PRIVATE "${cpp_httplib_SOURCE_DIR}")
if(WIN32)
  target_link_libraries(Isvik PRIVATE ws2_32)
endif()
target_compile_features(Isvik PRIVATE cxx_std_20)
slint_target_sources(Isvik ui/main_window.slint)

if(WIN32)
  enable_language(RC)
  target_sources(Isvik PRIVATE resources/Isvik.rc)
  set_property(SOURCE resources/Isvik.rc APPEND PROPERTY OBJECT_DEPENDS
    "${PROJECT_SOURCE_DIR}/resources/Isvik.ico"
    "${PROJECT_SOURCE_DIR}/resources/resource.h")
  target_include_directories(Isvik PRIVATE "${PROJECT_SOURCE_DIR}/resources")
  target_link_libraries(Isvik PRIVATE ole32 shell32 uuid bcrypt advapi32)
  add_custom_command(
    TARGET Isvik
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:Isvik> $<TARGET_FILE_DIR:Isvik>
    COMMAND_EXPAND_LISTS
    COMMENT "Copying runtime DLLs beside Isvik")
endif()

if(ISVIK_ENABLE_OPENVINO)
  if(ISVIK_ENABLE_OPENVINO_GENAI)
    include(cmake/IsvikOpenVinoGenAI.cmake)
    isvik_prepare_openvino_genai()
  endif()
  if(NOT OpenVINO_DIR AND WIN32)
    file(GLOB _isvik_openvino_cmake_dirs
      LIST_DIRECTORIES true
      "C:/Program Files (x86)/Intel/openvino_*/runtime/cmake")
    list(SORT _isvik_openvino_cmake_dirs)
    if(_isvik_openvino_cmake_dirs)
      list(REVERSE _isvik_openvino_cmake_dirs)
      list(GET _isvik_openvino_cmake_dirs 0 OpenVINO_DIR)
    endif()
  endif()
  if(DEFINED ENV{OPENVINO_DIR} AND NOT OpenVINO_DIR)
    set(OpenVINO_DIR "$ENV{OPENVINO_DIR}")
  endif()
  find_package(OpenVINO CONFIG REQUIRED)
  add_subdirectory(src/backends/openvino)
  target_link_libraries(Isvik PRIVATE isvik_openvino)
  target_compile_definitions(Isvik PRIVATE ISVIK_HAS_OPENVINO_RUNTIME=1)
  if(ISVIK_ENABLE_OPENVINO_GENAI)
    find_package(OpenVINOGenAI ${ISVIK_OPENVINO_GENAI_VERSION} EXACT CONFIG REQUIRED)
    target_link_libraries(isvik_openvino PUBLIC openvino::genai)
    target_compile_definitions(isvik_openvino PUBLIC ISVIK_HAS_OPENVINO_GENAI=1)
    file(TO_CMAKE_PATH "${ISVIK_OPENVINO_GENAI_ROOT}" _isvik_genai_root_for_cpp)
    target_compile_definitions(Isvik PRIVATE
      "ISVIK_OPENVINO_GENAI_ROOT=\"${_isvik_genai_root_for_cpp}\"")

    if(WIN32)
      add_custom_command(TARGET Isvik POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${ISVIK_OPENVINO_GENAI_ROOT}/runtime/bin/intel64/$<CONFIG>"
          "$<TARGET_FILE_DIR:Isvik>/openvino_bundle/runtime/bin/intel64/$<CONFIG>"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${ISVIK_OPENVINO_GENAI_ROOT}/runtime/bin/intel64/$<CONFIG>"
          "$<TARGET_FILE_DIR:Isvik>"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${ISVIK_OPENVINO_GENAI_ROOT}/runtime/3rdparty/tbb/bin"
          "$<TARGET_FILE_DIR:Isvik>/openvino_bundle/runtime/3rdparty/tbb/bin"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${ISVIK_OPENVINO_GENAI_ROOT}/runtime/3rdparty/tbb/bin"
          "$<TARGET_FILE_DIR:Isvik>"
        COMMENT "Packaging OpenVINO plugins and TBB runtime beside Isvik")
    endif()
  endif()
endif()

if(ISVIK_ENABLE_TENSORRT)
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH
      "TensorRT SDK root (directory containing include, lib, and bin)")
  if(NOT TENSORRT_ROOT)
    find_path(_isvik_tensorrt_include_dir NvInfer.h
      HINTS "C:/TensorRT/TensorRT-11.3.0.99" "C:/TensorRT"
      PATH_SUFFIXES include)
    if(_isvik_tensorrt_include_dir)
      get_filename_component(TENSORRT_ROOT "${_isvik_tensorrt_include_dir}" DIRECTORY)
      set(TENSORRT_ROOT "${TENSORRT_ROOT}" CACHE PATH
          "TensorRT SDK root (directory containing include, lib, and bin)" FORCE)
    endif()
  endif()

  find_package(CUDAToolkit QUIET)
  find_path(ISVIK_TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS "${TENSORRT_ROOT}/include")
  find_library(ISVIK_TENSORRT_LIBRARY NAMES nvinfer_11 nvinfer
    HINTS "${TENSORRT_ROOT}/lib")
  find_library(ISVIK_TENSORRT_PARSER_LIBRARY NAMES nvonnxparser_11 nvonnxparser
    HINTS "${TENSORRT_ROOT}/lib")
  find_library(ISVIK_TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin_11 nvinfer_plugin
    HINTS "${TENSORRT_ROOT}/lib")

  if(CUDAToolkit_FOUND AND ISVIK_TENSORRT_INCLUDE_DIR AND
     ISVIK_TENSORRT_LIBRARY AND ISVIK_TENSORRT_PARSER_LIBRARY AND
     ISVIK_TENSORRT_PLUGIN_LIBRARY)
    set(ISVIK_TENSORRT_AVAILABLE ON)
    enable_language(CUDA)
    set(ISVIK_TENSORRT_CUDA_ARCHITECTURES "120" CACHE STRING
        "CUDA architectures for Isvik's native GGUF TensorRT kernels")
    add_library(isvik_tensorrt
      src/backends/tensorrt/engine.cpp
      src/backends/tensorrt/gguf_genai_engine.cpp
      src/backends/tensorrt/gguf_matvec_plugin.cpp
      src/backends/tensorrt/gguf_matvec_kernel.cu)
    set_target_properties(isvik_tensorrt PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${ISVIK_TENSORRT_CUDA_ARCHITECTURES}")
    target_include_directories(isvik_tensorrt PUBLIC "${PROJECT_SOURCE_DIR}/include")
    target_include_directories(isvik_tensorrt PRIVATE "${ISVIK_TENSORRT_INCLUDE_DIR}")
    target_compile_features(isvik_tensorrt PUBLIC cxx_std_20)
    target_link_libraries(isvik_tensorrt
      PUBLIC isvik_core CUDA::cudart_static
      PRIVATE "${ISVIK_TENSORRT_LIBRARY}"
              "${ISVIK_TENSORRT_PARSER_LIBRARY}"
              "${ISVIK_TENSORRT_PLUGIN_LIBRARY}")
    if(MSVC)
      target_compile_options(isvik_tensorrt PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    else()
      target_compile_options(isvik_tensorrt PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    endif()
    target_link_libraries(Isvik PRIVATE isvik_tensorrt)
    target_compile_definitions(Isvik PRIVATE ISVIK_HAS_TENSORRT=1)

    if(WIN32)
      set(_isvik_tensorrt_dlls
        "${TENSORRT_ROOT}/bin/nvinfer_11.dll"
        "${TENSORRT_ROOT}/bin/nvinfer_plugin_11.dll"
        "${TENSORRT_ROOT}/bin/nvonnxparser_11.dll")
      foreach(_isvik_cuda_dll IN ITEMS cudart64_13.dll cublas64_13.dll cublasLt64_13.dll)
        unset(_isvik_cuda_runtime_dll CACHE)
        unset(_isvik_cuda_runtime_dll)
        find_file(_isvik_cuda_runtime_dll "${_isvik_cuda_dll}"
          HINTS "${CUDAToolkit_BIN_DIR}" "${CUDAToolkit_BIN_DIR}/x64")
        if(_isvik_cuda_runtime_dll)
          list(APPEND _isvik_tensorrt_dlls "${_isvik_cuda_runtime_dll}")
        endif()
      endforeach()
      file(GLOB _isvik_tensorrt_builder_dlls
        "${TENSORRT_ROOT}/bin/nvinfer_builder_resource_*.dll")
      list(APPEND _isvik_tensorrt_dlls ${_isvik_tensorrt_builder_dlls})
      foreach(_isvik_tensorrt_dll IN LISTS _isvik_tensorrt_dlls)
        if(EXISTS "${_isvik_tensorrt_dll}")
          add_custom_command(TARGET Isvik POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${_isvik_tensorrt_dll}" "$<TARGET_FILE_DIR:Isvik>"
            VERBATIM
            COMMENT "Deploying TensorRT runtime dependency ${_isvik_tensorrt_dll}")
        endif()
      endforeach()
      install(FILES ${_isvik_tensorrt_dlls} DESTINATION bin OPTIONAL)
    endif()

    message(STATUS "TensorRT backend enabled from ${TENSORRT_ROOT}")
    set(_isvik_tensorrt_config "${TENSORRT_ROOT}/cmake/TensorRT-EnterpriseConfig.cmake")
    if(EXISTS "${_isvik_tensorrt_config}")
      file(STRINGS "${_isvik_tensorrt_config}" _isvik_tensorrt_cuda_version_line
        REGEX "^set\\(TensorRT_CUDA_VERSION[ ]+")
      if(_isvik_tensorrt_cuda_version_line MATCHES "\"([0-9]+\\.[0-9]+)\"")
        set(_isvik_tensorrt_cuda_version "${CMAKE_MATCH_1}")
        set(_isvik_local_cuda_version
          "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
        if(NOT _isvik_tensorrt_cuda_version VERSION_EQUAL _isvik_local_cuda_version)
          message(WARNING "TensorRT SDK was built against CUDA ${_isvik_tensorrt_cuda_version}; "
                          "the selected toolkit is CUDA ${_isvik_local_cuda_version}. "
                          "The local TensorRT smoke test will verify this pairing.")
        endif()
      endif()
    endif()
  else()
    message(WARNING "ISVIK_ENABLE_TENSORRT is ON but TensorRT headers/libraries or CUDA Toolkit "
                    "were not found; building Isvik without TensorRT.")
  endif()
endif()

if(ISVIK_ENABLE_ONNXRUNTIME_GENAI)
  include(cmake/IsvikOnnxRuntimeGenAI.cmake)
  isvik_prepare_onnxruntime_genai()
  find_package(onnxruntime-genai 0.17.0 EXACT CONFIG REQUIRED
    PATHS "${ISVIK_ONNXRUNTIME_GENAI_ROOT}/lib/cmake/onnxruntime-genai"
    NO_DEFAULT_PATH)
  find_library(ISVIK_ONNXRUNTIME_LIBRARY NAMES onnxruntime
    PATHS "${ISVIK_ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  add_library(isvik_onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(isvik_onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ISVIK_ONNXRUNTIME_LIBRARY}"
    IMPORTED_LOCATION "${ISVIK_ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ISVIK_ONNXRUNTIME_ROOT}/include")

  add_library(isvik_onnxruntime_genai src/backends/onnxruntime/genai_engine.cpp)
  target_include_directories(isvik_onnxruntime_genai PUBLIC "${PROJECT_SOURCE_DIR}/include")
  target_compile_features(isvik_onnxruntime_genai PUBLIC cxx_std_20)
  target_link_libraries(isvik_onnxruntime_genai
    PUBLIC isvik_core
    PRIVATE onnxruntime-genai::onnxruntime-genai isvik_onnxruntime)
  if(MSVC)
    target_compile_options(isvik_onnxruntime_genai PRIVATE /W4 /permissive-)
  else()
    target_compile_options(isvik_onnxruntime_genai PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  endif()
  target_link_libraries(Isvik PRIVATE isvik_onnxruntime_genai)
  target_compile_definitions(Isvik PRIVATE ISVIK_HAS_ONNXRUNTIME_GENAI=1)

  file(GLOB _isvik_onnxruntime_runtime_dlls "${ISVIK_ONNXRUNTIME_ROOT}/lib/*.dll")
  file(GLOB _isvik_onnxruntime_genai_dlls "${ISVIK_ONNXRUNTIME_GENAI_ROOT}/lib/*.dll")
  if(WIN32)
    foreach(_isvik_runtime_dll IN LISTS _isvik_onnxruntime_runtime_dlls _isvik_onnxruntime_genai_dlls)
      add_custom_command(TARGET Isvik POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_isvik_runtime_dll}" "$<TARGET_FILE_DIR:Isvik>"
        VERBATIM
        COMMENT "Packaging ONNX Runtime dependency ${_isvik_runtime_dll}")
    endforeach()
  endif()
  install(FILES ${_isvik_onnxruntime_runtime_dlls} ${_isvik_onnxruntime_genai_dlls}
    DESTINATION bin)
  install(FILES
    "${ISVIK_ONNXRUNTIME_ROOT}/LICENSE"
    "${ISVIK_ONNXRUNTIME_ROOT}/ThirdPartyNotices.txt"
    "${ISVIK_ONNXRUNTIME_GENAI_ROOT}/LICENSE"
    "${ISVIK_ONNXRUNTIME_GENAI_ROOT}/ThirdPartyNotices.txt"
    DESTINATION licenses/onnxruntime-genai OPTIONAL)

  if(ISVIK_ENABLE_TENSORRT_RTX_EP AND ISVIK_TENSORRT_AVAILABLE)
    include(cmake/IsvikTensorRTRtx.cmake)
    isvik_prepare_tensorrt_rtx_ep()
    isvik_prepare_onnxruntime_genai_cuda()
    file(GLOB _isvik_tensorrt_rtx_ep_dlls "${ISVIK_TENSORRT_RTX_EP_ROOT}/*.dll")
    set(_isvik_onnxruntime_genai_cuda_dll
      "${ISVIK_ONNXRUNTIME_GENAI_CUDA_ROOT}/lib/onnxruntime-genai-cuda.dll")
    if(NOT EXISTS "${_isvik_onnxruntime_genai_cuda_dll}")
      message(FATAL_ERROR "ONNX Runtime GenAI CUDA add-on is missing: ${_isvik_onnxruntime_genai_cuda_dll}")
    endif()
    target_compile_definitions(isvik_onnxruntime_genai PRIVATE ISVIK_HAS_TENSORRT_RTX_EP=1)
    target_compile_definitions(Isvik PRIVATE ISVIK_HAS_TENSORRT_RTX_EP=1)
    foreach(_isvik_tensorrt_rtx_runtime_dll IN LISTS _isvik_tensorrt_rtx_ep_dlls)
      add_custom_command(TARGET Isvik POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_isvik_tensorrt_rtx_runtime_dll}" "$<TARGET_FILE_DIR:Isvik>"
        VERBATIM
        COMMENT "Packaging TensorRT-RTX runtime dependency ${_isvik_tensorrt_rtx_runtime_dll}")
    endforeach()
    add_custom_command(TARGET Isvik POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_isvik_onnxruntime_genai_cuda_dll}" "$<TARGET_FILE_DIR:Isvik>"
      VERBATIM
      COMMENT "Packaging ONNX Runtime GenAI CUDA device add-on")
    install(FILES ${_isvik_tensorrt_rtx_ep_dlls} "${_isvik_onnxruntime_genai_cuda_dll}"
      DESTINATION bin)
    install(FILES
      "${ISVIK_TENSORRT_RTX_EP_ROOT}/LICENSE"
      "${ISVIK_TENSORRT_RTX_EP_ROOT}/ThirdPartyNotices.txt"
      "${ISVIK_TENSORRT_RTX_EP_ROOT}/TRT_RTX_Acknowledgements.txt"
      DESTINATION licenses/tensorrt-rtx OPTIONAL)
  elseif(ISVIK_ENABLE_TENSORRT_RTX_EP)
    message(WARNING
      "TensorRT-RTX ONNX support needs both a working TensorRT build and ONNX Runtime GenAI; it will be disabled.")
  endif()
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Visual Studio 2026's CMake generator emits .slnx. Keep an equivalent legacy
# .sln alongside it for workflows that explicitly require the classic format.
if(CMAKE_GENERATOR STREQUAL "Visual Studio 18 2026")
  add_custom_target(IsvikSln ALL
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass
      -File "${PROJECT_SOURCE_DIR}/cmake/ConvertSlnxToSln.ps1"
      -InputPath "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.slnx"
      -OutputPath "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.sln"
    COMMAND powershell -NoProfile -ExecutionPolicy Bypass
      -File "${PROJECT_SOURCE_DIR}/cmake/ConvertSlnxToSln.ps1"
      -InputPath "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.slnx"
      -OutputPath "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}.sln"
    DEPENDS Isvik
    COMMENT "Synchronizing ${PROJECT_NAME}.sln in the project root and build directory when content changes")
endif()

install(TARGETS Isvik RUNTIME DESTINATION bin)
install(FILES
  "${ISVIK_SLINT_PREFIX}/licenses/LICENSE.md"
  "${ISVIK_SLINT_PREFIX}/licenses/THIRDPARTY.md"
  DESTINATION licenses)
install(FILES "${PROJECT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md" DESTINATION licenses)
install(FILES "${cpp_httplib_SOURCE_DIR}/LICENSE" DESTINATION licenses RENAME "cpp-httplib-LICENSE")
install(FILES "${nlohmann_json_SOURCE_DIR}/LICENSE.MIT"
  DESTINATION licenses
  RENAME "nlohmann-json-LICENSE.MIT")
if(WIN32)
  install(FILES "${ISVIK_SLINT_PREFIX}/lib/slint_cpp.dll" DESTINATION bin)
endif()

if(WIN32 AND ISVIK_ENABLE_OPENVINO AND ISVIK_ENABLE_OPENVINO_GENAI)
  foreach(_isvik_runtime_config IN ITEMS Debug Release)
    set(_isvik_runtime_dir
      "${ISVIK_OPENVINO_GENAI_ROOT}/runtime/bin/intel64/${_isvik_runtime_config}")
    install(DIRECTORY "${_isvik_runtime_dir}/"
      DESTINATION bin
      CONFIGURATIONS "${_isvik_runtime_config}")
    install(DIRECTORY "${_isvik_runtime_dir}/"
      DESTINATION "bin/openvino_bundle/runtime/bin/intel64/${_isvik_runtime_config}"
      CONFIGURATIONS "${_isvik_runtime_config}")
  endforeach()

  install(DIRECTORY "${ISVIK_OPENVINO_GENAI_ROOT}/runtime/3rdparty/tbb/bin/"
    DESTINATION bin
    CONFIGURATIONS Debug Release)
  install(DIRECTORY "${ISVIK_OPENVINO_GENAI_ROOT}/runtime/3rdparty/tbb/bin/"
    DESTINATION bin/openvino_bundle/runtime/3rdparty/tbb/bin
    CONFIGURATIONS Debug Release)
  install(DIRECTORY "${ISVIK_OPENVINO_GENAI_ROOT}/docs/licensing/"
    DESTINATION licenses/openvino-genai)
  install(FILES
    "${ISVIK_OPENVINO_GENAI_ROOT}/docs/openvino_tokenizers/LICENSE"
    "${ISVIK_OPENVINO_GENAI_ROOT}/docs/openvino_tokenizers/third-party-programs.txt"
    DESTINATION licenses/openvino-tokenizers)
  install(FILES
    "${ISVIK_OPENVINO_GENAI_ROOT}/runtime/3rdparty/tbb/TBB-LICENSE"
    DESTINATION licenses/oneTBB)
endif()
