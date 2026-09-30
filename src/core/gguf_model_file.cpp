#include "isvik/core/gguf_model_file.h"

#include "isvik/core/gguf_quant.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace isvik {

struct GgufModelFile::MappedFile {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0U;
#if defined(_WIN32)
  HANDLE mapping = nullptr;
  ~MappedFile() {
    if (data != nullptr) static_cast<void>(UnmapViewOfFile(data));
    if (mapping != nullptr) static_cast<void>(CloseHandle(mapping));
  }
#else
  ~MappedFile() {
    if (data != nullptr) static_cast<void>(munmap(const_cast<std::uint8_t*>(data), size));
  }
#endif
};

GgufModelFile::GgufModelFile(std::filesystem::path path,
                             GgufInspection inspection)
    : path_(std::move(path)),
      inspection_(std::move(inspection)),
      stream_(path_, std::ios::binary) {}

GgufModelFile::~GgufModelFile() = default;

void GgufModelFile::TryMapReadOnly() {
  if (inspection_.file_size == 0U ||
      inspection_.file_size > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return;
  }
  auto mapped = std::make_unique<MappedFile>();
  mapped->size = static_cast<std::size_t>(inspection_.file_size);
#if defined(_WIN32)
  HANDLE file = CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
  if (file == INVALID_HANDLE_VALUE) return;
  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0U, 0U, nullptr);
  static_cast<void>(CloseHandle(file));
  if (mapping == nullptr) return;
  const void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0U, 0U, 0U);
  if (view == nullptr) {
    static_cast<void>(CloseHandle(mapping));
    return;
  }
  mapped->mapping = mapping;
  mapped->data = static_cast<const std::uint8_t*>(view);
#else
  const int file = open(path_.c_str(), O_RDONLY);
  if (file < 0) return;
  void* view = mmap(nullptr, mapped->size, PROT_READ, MAP_PRIVATE, file, 0);
  static_cast<void>(close(file));
  if (view == MAP_FAILED) return;
  mapped->data = static_cast<const std::uint8_t*>(view);
#endif
  mapped_file_ = std::move(mapped);
}

StatusOr<std::shared_ptr<GgufModelFile>> GgufModelFile::Open(
    const std::filesystem::path& path) {
  StatusOr<GgufInspection> inspection = GgufInspector::Inspect(path);
  if (!inspection.ok()) return inspection.status();

  auto model = std::shared_ptr<GgufModelFile>(
      new GgufModelFile(path, std::move(inspection).value()));
  if (!model->stream_.is_open()) {
    return Status::Unavailable("cannot open GGUF model for read-only tensor access");
  }

  std::error_code error;
  const uint64_t current_size = std::filesystem::file_size(path, error);
  if (error) {
    return Status::Unavailable("cannot verify GGUF model size: " + error.message());
  }
  if (current_size != model->inspection_.file_size) {
    return Status::Unavailable("GGUF model changed while it was being opened");
  }
  model->TryMapReadOnly();
  return model;
}

StatusOr<std::vector<uint8_t>> GgufModelFile::ReadTensorRange(
    std::string_view tensor_name, uint64_t offset, uint64_t size) const {
  const auto tensor = std::find_if(
      inspection_.tensors.begin(), inspection_.tensors.end(),
      [tensor_name](const GgufTensorInfo& candidate) {
        return candidate.name == tensor_name;
      });
  if (tensor == inspection_.tensors.end()) {
    return Status::NotFound("GGUF tensor does not exist: " + std::string(tensor_name));
  }
  if (!tensor->payload_bytes.has_value()) {
    return Status::Unsupported("GGUF tensor payload size is unknown: " +
                               std::string(tensor_name));
  }
  const uint64_t payload_size = tensor->payload_bytes.value();
  if (offset > payload_size || size > payload_size - offset) {
    return Status::InvalidArgument("GGUF tensor read exceeds its payload: " +
                                   std::string(tensor_name));
  }
  if (size > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return Status::InvalidArgument("GGUF tensor read is too large");
  }
  if (tensor->data_offset > std::numeric_limits<uint64_t>::max() - offset) {
    return Status::InvalidArgument("GGUF tensor file offset overflows");
  }
  const uint64_t absolute_offset = tensor->data_offset + offset;
  if (absolute_offset >
      static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    return Status::InvalidArgument("GGUF tensor file offset is too large");
  }
  if (size == 0U) return std::vector<uint8_t>{};

  std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
  if (mapped_file_ != nullptr) {
    std::memcpy(bytes.data(), mapped_file_->data + static_cast<std::size_t>(absolute_offset),
                bytes.size());
    return bytes;
  }
  std::lock_guard lock(mutex_);
  stream_.clear();
  stream_.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
  if (!stream_) {
    return Status::Unavailable("cannot seek to GGUF tensor: " +
                               std::string(tensor_name));
  }
  stream_.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(size));
  if (!stream_ || stream_.gcount() != static_cast<std::streamsize>(size)) {
    return Status::Unavailable("cannot read complete GGUF tensor range: " +
                               std::string(tensor_name));
  }
  return bytes;
}

StatusOr<std::vector<float>> GgufModelFile::ReadDecodedTensorBlocks(
    std::string_view tensor_name, uint64_t first_block,
    uint64_t block_count) const {
  const auto tensor = std::find_if(
      inspection_.tensors.begin(), inspection_.tensors.end(),
      [tensor_name](const GgufTensorInfo& candidate) {
        return candidate.name == tensor_name;
      });
  if (tensor == inspection_.tensors.end()) {
    return Status::NotFound("GGUF tensor does not exist: " + std::string(tensor_name));
  }
  const auto block = GgufQuantBlockInfoForType(tensor->type);
  if (!block.has_value()) {
    return Status::Unsupported("GGUF tensor encoding has no known block layout: " +
                               std::string(tensor_name));
  }
  if (!IsGgufTensorEncodingDecodable(tensor->type)) {
    return Status::Unsupported("GGUF tensor encoding has no native block decoder: " +
                               std::string(tensor_name));
  }
  if (!tensor->payload_bytes.has_value()) {
    return Status::Unsupported("GGUF tensor payload size is unknown: " +
                               std::string(tensor_name));
  }
  if (first_block > std::numeric_limits<uint64_t>::max() / block->bytes ||
      block_count > std::numeric_limits<uint64_t>::max() / block->bytes) {
    return Status::InvalidArgument("GGUF block range overflows: " +
                                   std::string(tensor_name));
  }
  if (block_count > kMaxGgufCpuDecodeElements / block->elements) {
    return Status::InvalidArgument("decoded GGUF block range exceeds the bounded output size: " +
                                   std::string(tensor_name));
  }
  const uint64_t offset = first_block * block->bytes;
  const uint64_t byte_count = block_count * block->bytes;
  auto source = ReadTensorRange(tensor_name, offset, byte_count);
  if (!source.ok()) return source.status();
  return DecodeGgufTensorBlocks(tensor->type, source.value());
}

}  // namespace isvik
