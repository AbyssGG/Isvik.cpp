#ifndef ISVIK_APP_API_SERVER_H_
#define ISVIK_APP_API_SERVER_H_

#include <functional>
#include <memory>
#include <string>

#include "isvik/core/cancellation.h"
#include "isvik/core/inference_event.h"
#include "isvik/core/inference_request.h"
#include "isvik/core/model_descriptor.h"
#include "isvik/core/status.h"

namespace isvik::app {

struct ApiServerConfig {
  std::string host = "127.0.0.1";
  int port = 1234;
  std::string api_key;
  std::string backend = "OpenVINO";
};

using ApiGenerate = std::function<Status(
    const UnifiedInferenceRequest&, const CancellationToken&,
    const std::function<void(const InferenceEvent&)>&)>;

class ApiServer {
 public:
  ApiServer();
  ~ApiServer();

  ApiServer(const ApiServer&) = delete;
  ApiServer& operator=(const ApiServer&) = delete;

  [[nodiscard]] Status Start(const ApiServerConfig& config,
                             const ModelDescriptor& model,
                             ApiGenerate generate);
  void Stop();
  void Wait();
  [[nodiscard]] bool IsRunning() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

int RunApiServer(const ApiServerConfig& config, const ModelDescriptor& model,
                 const ApiGenerate& generate);

}  // namespace isvik::app

#endif  // ISVIK_APP_API_SERVER_H_
