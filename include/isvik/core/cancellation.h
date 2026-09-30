#ifndef ISVIK_CORE_CANCELLATION_H_
#define ISVIK_CORE_CANCELLATION_H_

#include <atomic>
#include <memory>
#include <utility>

namespace isvik {

class CancellationSource;

class CancellationToken {
 public:
  CancellationToken() = default;

  [[nodiscard]] bool IsCancellationRequested() const {
    return state_ != nullptr && state_->cancelled.load(std::memory_order_acquire);
  }

 private:
  struct State {
    std::atomic_bool cancelled = false;
  };

  explicit CancellationToken(std::shared_ptr<State> state) : state_(std::move(state)) {}

  std::shared_ptr<State> state_;
  friend class CancellationSource;
};

class CancellationSource {
 public:
  CancellationSource() : state_(std::make_shared<CancellationToken::State>()) {}

  [[nodiscard]] CancellationToken token() const { return CancellationToken(state_); }

  void Cancel() const { state_->cancelled.store(true, std::memory_order_release); }

 private:
  std::shared_ptr<CancellationToken::State> state_;
};

}  // namespace isvik

#endif  // ISVIK_CORE_CANCELLATION_H_
