#include <cerrno>
#include <csignal>
#include <cstdint>
#include <fcntl.h>
#include <poll.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include "jaccl/tcp.h"

static int64_t now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

static jaccl::address_t bound_address(int fd) {
  jaccl::address_t addr{};
  addr.len = sizeof(addr.addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr.addr), &addr.len) < 0)
    throw std::runtime_error("getsockname failed");
  return addr;
}

static std::pair<jaccl::TCPSocket, jaccl::TCPSocket> connected_pair() {
  jaccl::TCPSocket server("[t]");
  server.listen("[t]", jaccl::parse_address("127.0.0.1:0"));
  auto client = jaccl::TCPSocket::connect("[t]", bound_address(server));
  auto peer = server.accept("[t]");
  return {std::move(client), std::move(peer)};
}

static int fails = 0;
static void check(bool ok, const char* what, int64_t took) {
  std::printf("%s %s (%lld ms)\n", ok ? "ok  " : "FAIL", what, static_cast<long long>(took));
  if (!ok) fails++;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  alarm(15);
  using namespace jaccl;
  {
    auto addr = parse_address("127.0.0.1:0");
    TCPSocket server("[t]");
    server.listen("[t]", addr);
    jaccl_set_formation_deadline_ms(now_ms() + 300);
    int64_t t0 = now_ms();
    bool threw = false;
    try {
      server.accept("[t]");
    } catch (std::runtime_error& e) {
      threw = std::string(e.what()) == "[t] formation deadline passed in accept";
    }
    int64_t took = now_ms() - t0;
    check(threw && took >= 200 && took < 1500, "accept with no peer throws at the deadline", took);
  }
  {
    auto addr = [] {
      TCPSocket reserved("[t]");
      reserved.listen("[t]", parse_address("127.0.0.1:0"));
      return bound_address(reserved);
    }();
    jaccl_set_formation_deadline_ms(now_ms() + 400);
    int64_t t0 = now_ms();
    bool threw = false;
    try {
      TCPSocket::connect("[t]", addr, 20, 20, nullptr);
    } catch (std::runtime_error& e) {
      int attempts = 0;
      char trailing = 0;
      threw = std::sscanf(e.what(),
          "[t] formation deadline passed in connect after %d attempts%c",
          &attempts, &trailing) == 1 && attempts > 0 && attempts < 20;
    }
    int64_t took = now_ms() - t0;
    check(threw && took >= 250 && took < 1500, "connect to nobody stops at the deadline, not after 20 attempts", took);
  }
  {
    jaccl_set_formation_deadline_ms(0);
    auto addr = parse_address("127.0.0.1:0");
    TCPSocket server("[t]");
    server.listen("[t]", addr);
    addr = bound_address(server);
    std::thread c([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      auto s = TCPSocket::connect("[t]", addr, 4, 20, nullptr);
      int x = 7;
      s.send("[t]", &x, sizeof(x));
    });
    int64_t t0 = now_ms();
    auto peer = server.accept("[t]");
    int got = 0;
    peer.recv("[t]", &got, sizeof(got));
    c.join();
    int64_t took = now_ms() - t0;
    check(got == 7 && took >= 400, "with no deadline, accept waits as before and the exchange works", took);
  }
  {
    auto sockets = connected_pair();
    jaccl_set_formation_deadline_ms(now_ms() + 300);
    int64_t t0 = now_ms();
    int got = 0;
    bool threw = false;
    try {
      sockets.second.recv("[t]", &got, sizeof(got));
    } catch (std::runtime_error& e) {
      threw = std::string(e.what()) == "[t] formation deadline passed in recv";
    }
    int64_t took = now_ms() - t0;
    check(threw && took >= 200 && took < 1500, "recv from a silent peer stops at the deadline", took);
    jaccl_set_formation_deadline_ms(0);
  }
  {
    auto sockets = connected_pair();
    int fd = sockets.first;
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
      throw std::runtime_error("set nonblocking failed");
    char data[65536] = {};
    for (;;) {
      auto n = ::send(fd, data, sizeof(data), 0);
      if (n > 0) continue;
      if (n < 0 && errno == EINTR) continue;
      if (n >= 0 || (errno != EAGAIN && errno != EWOULDBLOCK))
        throw std::runtime_error("fill socket failed");
      pollfd p = {fd, POLLOUT, 0};
      if (poll(&p, 1, 20) == 0) break;
    }
    if (fcntl(fd, F_SETFL, flags) < 0)
      throw std::runtime_error("restore blocking failed");
    jaccl_set_formation_deadline_ms(now_ms() + 300);
    int64_t t0 = now_ms();
    bool threw = false;
    try {
      sockets.first.send("[t]", data, 1);
    } catch (std::runtime_error& e) {
      threw = std::string(e.what()) == "[t] formation deadline passed in send";
    }
    int64_t took = now_ms() - t0;
    check(threw && took >= 200 && took < 1500, "send to a full socket stops at the deadline", took);
    jaccl_set_formation_deadline_ms(0);
  }
  std::printf("JACCL-DEADLINE-BEHAVIOR %s\n", fails ? "FAIL" : "OK");
  return fails ? 1 : 0;
}
