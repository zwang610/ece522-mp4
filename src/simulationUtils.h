#ifndef __SIMULATION_UTILS_H__
#define __SIMULATION_UTILS_H__

#include <string>
#include <cstdint>
#include "analysis.h"

using std::string;

#define PAGE_SIZE (4096)
typedef uint64_t Addr;

inline bool isPageAligned(Addr addr) {
  return addr % PAGE_SIZE == 0;
}

inline bool isPageSized(unsigned long size) {
  return size % PAGE_SIZE == 0;
}

namespace Simulator {

enum TensorLocation{ NOT_PRESENT, IN_SSD, IN_CPU, IN_GPU, NOT_KNOWN, IN_GPU_LEAST };

const std::string print_pagelocation_array [6] = {
    "Not_Present", "In_SSD", "In_CPU", "In_GPU", "Not_Known", "In_GPU_Least"
};

enum GPUPageTableEvcPolicy{ RANDOM, LRU, GUIDED };

enum MigPolicy{ DEEPUM, OURS };

/**
 * @brief
 */
class TensorMovementHint {
  public:
    TensorMovementHint(TensorLocation from, TensorLocation to,
                     int issued_kernel_id, Tensor* tensor) :
        from(from), to(to), issued_kernel_id(issued_kernel_id), tensor(tensor) {
      assert(to != NOT_KNOWN);
    }
    bool operator<(const TensorMovementHint& rhs) const {
      return issued_kernel_id < rhs.issued_kernel_id;
    }

    TensorLocation from;
    TensorLocation to;
    string human_readable_hint;
    int issued_kernel_id;
    Tensor* tensor;
};

} // namespace Simulator

#endif
