#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "mlx/c/distributed_group.h"
#include "mlx/c/error.h"
#include "mlx/c/private/distributed_group.h"
#include "mlx/distributed/distributed_impl.h"

namespace mx = mlx::core;

static std::string last_error;

static void capture_error(const char* message, void*) {
  last_error = message;
}

static void require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s (MLX: %s)\n", message, last_error.c_str());
    std::exit(1);
  }
}

class BoundaryGroup : public mx::distributed::detail::GroupImpl {
 public:
  std::string reason;
  bool fail_progress = false;

  void abort(const std::string& value) override {
    reason = value;
    if (value == "failure %s %n") {
      throw std::runtime_error(value);
    }
  }
  uint64_t progress() const override {
    if (fail_progress) {
      throw std::runtime_error("progress failure %s %n");
    }
    return UINT64_C(0x123456789abcdef0);
  }
  int rank() override {
    return 0;
  }
  int size() override {
    return 1;
  }
  mx::Stream communication_stream(mx::StreamOrDevice) override {
    throw std::runtime_error("unexpected communication");
  }
  std::shared_ptr<mx::distributed::detail::GroupImpl> split(int, int) override {
    throw std::runtime_error("unexpected split");
  }
  void all_sum(const mx::array&, mx::array&, mx::Stream) override {}
  void all_gather(const mx::array&, mx::array&, mx::Stream) override {}
  void send(const mx::array&, int, mx::Stream) override {}
  void recv(mx::array&, int, mx::Stream) override {}
  void all_max(const mx::array&, mx::array&, mx::Stream) override {}
  void all_min(const mx::array&, mx::array&, mx::Stream) override {}
  void sum_scatter(const mx::array&, mx::array&, mx::Stream) override {}
};

int main() {
  mlx_set_error_handler(capture_error, nullptr, nullptr);
  auto empty = mlx_distributed_group_new();
  uint64_t output = 42;
  require(mlx_distributed_group_abort(empty, "abort") == 1, "empty abort");
  require(last_error.find("non-empty") != std::string::npos, "empty error");
  require(
      mlx_distributed_group_progress(empty, &output) == 1, "empty progress");
  require(output == 42, "failed progress preserves output");
  require(mlx_distributed_group_free(empty) == 0, "free empty group");

  auto impl = std::make_shared<BoundaryGroup>();
  auto group = mlx_distributed_group_new_(mx::distributed::Group(impl));
  require(mlx_distributed_group_abort(group, nullptr) == 1, "null reason");
  require(last_error.find("reason") != std::string::npos, "null reason error");
  require(impl->reason.empty(), "null reason never reaches backend");
  require(mlx_distributed_group_progress(group, nullptr) == 1, "null output");
  require(last_error.find("output") != std::string::npos, "null output error");
  require(
      mlx_distributed_group_abort(group, "caller reason") == 0,
      "abort success");
  require(impl->reason == "caller reason", "abort forwards reason");
  require(
      mlx_distributed_group_progress(group, &output) == 0, "progress success");
  require(output == UINT64_C(0x123456789abcdef0), "progress preserves 64 bits");
  require(
      mlx_distributed_group_abort(group, "failure %s %n") == 1,
      "abort exception");
  require(
      last_error.find("failure %s %n") != std::string::npos,
      "literal abort error");
  impl->fail_progress = true;
  require(
      mlx_distributed_group_progress(group, &output) == 1,
      "progress exception");
  require(output == UINT64_C(0x123456789abcdef0), "exception preserves output");
  require(
      last_error.find("progress failure %s %n") != std::string::npos,
      "literal progress error");
  require(mlx_distributed_group_free(group) == 0, "free boundary group");

  unsetenv("MLX_HOSTFILE");
  unsetenv("MLX_RANK");
  require(mlx_distributed_clear_backends() == 0, "clear backend cache");
  auto singleton = mlx_distributed_group_new();
  require(
      mlx_distributed_init(&singleton, false, "ring") == 0, "init singleton");
  require(
      mlx_distributed_group_abort(singleton, "unsupported") == 1,
      "unsupported abort");
  require(
      last_error.find("JACCL") != std::string::npos, "unsupported abort error");
  require(
      mlx_distributed_group_progress(singleton, &output) == 1,
      "unsupported progress");
  require(
      last_error.find("JACCL") != std::string::npos,
      "unsupported progress error");
  require(mlx_distributed_group_free(singleton) == 0, "free singleton");
  require(mlx_distributed_clear_backends() == 0, "clear before reinit");
  singleton = mlx_distributed_group_new();
  require(
      mlx_distributed_init(&singleton, false, "ring") == 0, "reinit singleton");
  require(
      mlx_distributed_group_size(singleton) == 1, "reopened singleton size");
  require(
      mlx_distributed_group_free(singleton) == 0, "free reopened singleton");
  require(mlx_distributed_clear_backends() == 0, "clear after teardown");
  std::puts("PASS: distributed group C ABI");
}
