#include "isvik/core/status.h"

#include <utility>

namespace isvik {

Status::Status(StatusCode code, std::string message)
    : code_(code), message_(std::move(message)) {
  if (code_ == StatusCode::kOk) {
    message_.clear();
  }
}

Status Status::Cancelled(std::string message) {
  return Status(StatusCode::kCancelled, std::move(message));
}

Status Status::InvalidArgument(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status Status::NotFound(std::string message) {
  return Status(StatusCode::kNotFound, std::move(message));
}

Status Status::AlreadyExists(std::string message) {
  return Status(StatusCode::kAlreadyExists, std::move(message));
}

Status Status::Unsupported(std::string message) {
  return Status(StatusCode::kUnsupported, std::move(message));
}

Status Status::Unavailable(std::string message) {
  return Status(StatusCode::kUnavailable, std::move(message));
}

Status Status::Internal(std::string message) {
  return Status(StatusCode::kInternal, std::move(message));
}

}  // namespace isvik
