// Diagnostic only: retain command buffers, read public timestamps after drain.
// No completion callbacks, inserted GPU commands, per-kernel hooks or MLX edits.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <mach/mach_time.h>
#include <os/lock.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
  uint64_t sequence;
  double cpu_commit, gpu_start, gpu_end, cpu_complete, driver_start, driver_end;
  uint64_t status;
} Row;
static os_unfair_lock lock_ = OS_UNFAIR_LOCK_INIT;
static Row* rows_ = NULL;
static CFTypeRef* buffers_ = NULL;
static int count_ = 0, overflow_ = 0, enabled_ = 0;
static const int capacity_ = 20000;
static Class target_ = Nil;
static IMP original_ = NULL;
static double seconds_per_tick_ = 0;
static double now_(void) { return mach_absolute_time() * seconds_per_tick_; }

static void commit_(id object, SEL selector) {
  if (__atomic_load_n(&enabled_, __ATOMIC_RELAXED)) {
    os_unfair_lock_lock(&lock_);
    if (count_ < capacity_) {
      rows_[count_] = (Row){.sequence=(uint64_t)count_, .cpu_commit=now_()};
      buffers_[count_] = CFRetain((__bridge CFTypeRef)object);
      count_++;
    } else overflow_ = 1;
    os_unfair_lock_unlock(&lock_);
  }
  ((void (*)(id, SEL))original_)(object, selector);
}

const char* kms_install(void) {
  if (target_ != Nil) return class_getName(target_);
  mach_timebase_info_data_t base;
  mach_timebase_info(&base);
  seconds_per_tick_ = (double)base.numer / (double)base.denom / 1e9;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return NULL;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBufferWithUnretainedReferences];
    target_ = object_getClass(buffer);
    Method method = class_getInstanceMethod(target_, @selector(commit));
    original_ = method_getImplementation(method);
    if (!class_addMethod(target_, @selector(commit), (IMP)commit_, method_getTypeEncoding(method)))
      class_replaceMethod(target_, @selector(commit), (IMP)commit_, method_getTypeEncoding(method));
    rows_ = calloc(capacity_, sizeof(Row));
    buffers_ = calloc(capacity_, sizeof(CFTypeRef));
    if (!rows_ || !buffers_) abort();
    return class_getName(target_);
  }
}
int kms_begin(void) {
  if (target_ == Nil || enabled_) return -1;
  for (int i=0; i<count_; i++) if (buffers_[i] != NULL) return -1;
  count_=0; overflow_=0;
  __atomic_store_n(&enabled_, 1, __ATOMIC_RELEASE);
  return 0;
}
int kms_end(int timeout_ms) {
  __atomic_store_n(&enabled_, 0, __ATOMIC_RELEASE);
  os_unfair_lock_lock(&lock_); os_unfair_lock_unlock(&lock_);
  double deadline = now_() + timeout_ms/1000.;
  int failed = overflow_;
  for (int i=0; i<count_; i++) {
    if (buffers_[i] == NULL) continue;
    id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)buffers_[i];
    while (buffer.status < MTLCommandBufferStatusCompleted && now_() < deadline) usleep(1000);
    rows_[i].gpu_start=buffer.GPUStartTime; rows_[i].gpu_end=buffer.GPUEndTime;
    rows_[i].cpu_complete=now_(); rows_[i].driver_start=buffer.kernelStartTime;
    rows_[i].driver_end=buffer.kernelEndTime; rows_[i].status=buffer.status;
    if (buffer.status != MTLCommandBufferStatusCompleted) failed=1;
    CFRelease(buffers_[i]); buffers_[i]=NULL;
  }
  return failed ? -1 : count_;
}
int kms_rows(Row* output, int capacity) {
  int n = capacity < count_ ? capacity : count_;
  memcpy(output, rows_, (size_t)n*sizeof(Row));
  return n;
}
int kms_uninstall(void) {
  if (enabled_) return -1;
  for (int i=0; i<count_; i++) if (buffers_[i] != NULL) return -1;
  if (target_ != Nil) {
    Method method=class_getInstanceMethod(target_, @selector(commit));
    class_replaceMethod(target_, @selector(commit), original_, method_getTypeEncoding(method));
    target_=Nil; free(rows_); free(buffers_); rows_=NULL; buffers_=NULL; count_=0;
  }
  return 0;
}
