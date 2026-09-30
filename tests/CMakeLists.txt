include(FetchContent)

set(BUILD_GMOCK OFF CACHE BOOL "Build GoogleMock with the project tests" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest as part of Isvik" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "Use the MSVC shared C runtime" FORCE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.18.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(googletest)

add_executable(isvik_core_tests core_test.cpp)
target_link_libraries(isvik_core_tests PRIVATE isvik_core GTest::gtest_main)
target_compile_features(isvik_core_tests PRIVATE cxx_std_20)

add_executable(isvik_feature_tests core_features_test.cpp)
target_link_libraries(isvik_feature_tests PRIVATE isvik_core GTest::gtest_main)
target_compile_features(isvik_feature_tests PRIVATE cxx_std_20)

add_executable(isvik_memory_tests memory_service_test.cpp)
target_link_libraries(isvik_memory_tests PRIVATE
  isvik_core isvik_sqlite GTest::gtest_main)
target_compile_features(isvik_memory_tests PRIVATE cxx_std_20)

add_executable(isvik_catalog_tests model_catalog_test.cpp)
target_link_libraries(isvik_catalog_tests PRIVATE isvik_core GTest::gtest_main)
target_compile_features(isvik_catalog_tests PRIVATE cxx_std_20)

if(TARGET isvik_openvino)
  target_compile_definitions(isvik_core_tests PRIVATE ISVIK_TEST_OPENVINO_BACKEND=1)

  if(WIN32)
    add_executable(isvik_openvino_tests openvino_runtime_test.cpp)
    target_link_libraries(isvik_openvino_tests PRIVATE isvik_openvino GTest::gtest_main)
    target_compile_features(isvik_openvino_tests PRIVATE cxx_std_20)
    add_test(NAME OpenVinoRuntimeTest.EnumeratesTheInstalledCpuPlugin
      COMMAND isvik_openvino_tests)

    get_filename_component(_isvik_openvino_runtime_root "${OpenVINO_DIR}/.." ABSOLUTE)
    set(_isvik_openvino_test_path
      "${_isvik_openvino_runtime_root}/bin/intel64/Release;${_isvik_openvino_runtime_root}/bin/intel64/Debug;${_isvik_openvino_runtime_root}/3rdparty/tbb/bin;$ENV{PATH}")
    set(_isvik_openvino_library_paths
      "${_isvik_openvino_runtime_root}/bin/intel64/Release;${_isvik_openvino_runtime_root}/bin/intel64/Debug;${_isvik_openvino_runtime_root}/3rdparty/tbb/bin")
    string(REPLACE ";" "\\;" _isvik_openvino_test_path "${_isvik_openvino_test_path}")
    string(REPLACE ";" "\\;" _isvik_openvino_library_paths "${_isvik_openvino_library_paths}")
    set_tests_properties(OpenVinoRuntimeTest.EnumeratesTheInstalledCpuPlugin PROPERTIES
      ENVIRONMENT
        "PATH=${_isvik_openvino_test_path};OPENVINO_LIB_PATHS=${_isvik_openvino_library_paths};INTEL_OPENVINO_DIR=${_isvik_openvino_runtime_root}/..")
  endif()
endif()

if(TARGET isvik_tensorrt)
  add_executable(isvik_tensorrt_tests tensorrt_runtime_test.cpp)
  target_link_libraries(isvik_tensorrt_tests PRIVATE isvik_tensorrt GTest::gtest_main)
  target_compile_features(isvik_tensorrt_tests PRIVATE cxx_std_20)
  add_test(NAME TensorRtRuntimeTest.BuildsAndExecutesOnNvidiaGpu
    COMMAND isvik_tensorrt_tests)
  if(WIN32)
    set_tests_properties(TensorRtRuntimeTest.BuildsAndExecutesOnNvidiaGpu PROPERTIES
      ENVIRONMENT "PATH=${TENSORRT_ROOT}/bin\;${CUDAToolkit_BIN_DIR}\;$ENV{PATH}")
  endif()
endif()

include(GoogleTest)
gtest_discover_tests(isvik_core_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(isvik_feature_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(isvik_memory_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(isvik_catalog_tests DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
