// Test guard: cap table mappings and trap any access to table bytes.
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

static size_t windows;

static int is_table(int fd) {
    char path[4096];
    return fd >= 0 && fcntl(fd, F_GETPATH, path) == 0 && strstr(path, "/engram/");
}

static void *guard_map(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!is_table(fd)) return mmap(addr, len, prot, flags, fd, off);
    if (len > (64UL << 20) || prot != PROT_READ) _exit(86);
    ++windows;
    return mmap(addr, len, PROT_NONE, flags, fd, off);
}

static ssize_t guard_read(int fd, void *buf, size_t len) {
    if (is_table(fd)) _exit(87);
    return read(fd, buf, len);
}

static ssize_t guard_pread(int fd, void *buf, size_t len, off_t off) {
    if (is_table(fd)) _exit(88);
    return pread(fd, buf, len, off);
}

static ssize_t guard_preadv(int fd, const struct iovec *vec, int count, off_t off) {
    if (is_table(fd)) _exit(89);
    return preadv(fd, vec, count, off);
}

static ssize_t guard_readv(int fd, const struct iovec *vec, int count) {
    if (is_table(fd)) _exit(90);
    return readv(fd, vec, count);
}

__attribute__((destructor)) static void report(void) {
    fprintf(stderr, "PP4-GUARD windows=%zu\n", windows);
}

#define INTERPOSE(replacement, original) \
    __attribute__((used, section("__DATA,__interpose"))) \
    static const struct { const void *new_fn; const void *old_fn; } pair_##original = \
        { (const void *)(replacement), (const void *)(original) }

INTERPOSE(guard_map, mmap);
INTERPOSE(guard_read, read);
INTERPOSE(guard_pread, pread);
INTERPOSE(guard_preadv, preadv);
INTERPOSE(guard_readv, readv);
