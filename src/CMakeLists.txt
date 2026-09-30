add_library(isvik_core
  core/chat_context.cpp
  core/context_manager.cpp
  core/generation_config.cpp
  core/gguf_model_file.cpp
  core/gguf_inspector.cpp
  core/gguf_quant.cpp
  core/gguf_tokenizer.cpp
  core/inference_request.cpp
  core/model_manager.cpp
  core/model_catalog.cpp
  core/model_registry.cpp
  core/memory_service.cpp
  core/status.cpp
  core/tool_runtime.cpp
  i18n/localization.cpp)

target_include_directories(isvik_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(isvik_core
  PUBLIC nlohmann_json::nlohmann_json
  PRIVATE isvik_sqlite)
target_compile_features(isvik_core PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(isvik_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(isvik_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()
