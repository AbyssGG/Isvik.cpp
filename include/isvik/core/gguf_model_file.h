#ifndef ISVIK_CORE_GGUF_MODEL_FILE_H_
#define ISVIK_CORE_GGUF_MODEL_FILE_H_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

#include "isvik/core/gguf_inspector.h"

namespace isvik {

// Opens a GGUF read-only and serves bounded raw tensor slices from the original
// file. The reader does not convert, rewrite, or copy the model as a whole.
class GgufModelFile {
 public:
  [[nodiscard]] static StatusOr<std::shared_ptr<GgufModelFile>> Open(
      const std::filesystem::path& path);
  ~GgufModelFile();

  [[nodiscard]] const GgufInspection& inspection() const { return inspection_; }

  // Offsets and sizes are relative to the tensor payload, not the GGUF file.
  // Unknown tensor encodings can be inspected, but cannot be safely read until
  // their payload size is known.
  [[nodiscard]] StatusOr<std::vector<uint8_t>> ReadTensorRange(
      std::string_view tensor_name, uint64_t offset, uint64_t size) const;

  // Reads and decodes only the requested complete blocks from the original
  // file. This bounded CPU path is a correctness reference for backend kernels.
  [[nodiscard]] StatusOr<std::vector<float>> ReadDecodedTensorBlocks(
      std::string_view tensor_name, uint64_t first_block,
      uint64_t block_count) const;

 private:
  struct MappedFile;
  GgufModelFile(std::filesystem::path path, GgufInspection inspection);
  void TryMapReadOnly();

  std::filesystem::path path_;
  GgufInspection inspection_;
  std::unique_ptr<MappedFile> mapped_file_;
  mutable std::mutex mutex_;
  mutable std::ifstream stream_;
};

}  // namespace isvik

#endif  // ISVIK_CORE_GGUF_MODEL_FILE_H_
