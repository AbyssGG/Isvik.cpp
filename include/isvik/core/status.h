#ifndef ISVIK_CORE_STATUS_H_
#define ISVIK_CORE_STATUS_H_

#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace isvik {

enum class StatusCode {
  kOk = 0,
  kCancelled,
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kUnsupported,
  kUnavailable,
  kInternal,
};

class Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message);

  [[nodiscard]] bool ok() const { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const { return code_; }
  [[nodiscard]] const std::string& message() const { return message_; }

  static Status Cancelled(std::string message);
  static Status InvalidArgument(std::string message);
  static Status NotFound(std::string message);
  static Status AlreadyExists(std::string message);
  static Status Unsupported(std::string message);
  static Status Unavailable(std::string message);
  static Status Internal(std::string message);

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

template <typename T>
class StatusOr {
  static_assert(!std::is_reference_v<T>, "StatusOr cannot hold references");

 public:
  StatusOr(T value) : value_(std::move(value)) {}
  StatusOr(Status status) : status_(std::move(status)) {
    if (status_.ok()) {
      status_ = Status::Internal("StatusOr cannot contain an OK status without a value");
    }
  }

  [[nodiscard]] bool ok() const { return value_.has_value(); }
  [[nodiscard]] const Status& status() const { return status_; }

  const T& value() const& {
    EnsureValue();
    return *value_;
  }

  T& value() & {
    EnsureValue();
    return *value_;
  }

  T&& value() && {
    EnsureValue();
    return std::move(*value_);
  }

 private:
  void EnsureValue() const {
    if (!value_) {
      throw std::logic_error(status_.message());
    }
  }

  Status status_;
  std::optional<T> value_;
};

}  // namespace isvik

#endif  // ISVIK_CORE_STATUS_H_
