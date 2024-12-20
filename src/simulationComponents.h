#ifndef __SIMULATION_COMPONENTS_H__
#define __SIMULATION_COMPONENTS_H__

#include <queue>
#include <set>
#include <map>
#include <list>
#include <deque>
#include <vector>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include "ast.h"
#include "analysis.h"
#include "simulationUtils.h"

using std::priority_queue;
using std::set;
using std::map;
using std::pair;
using std::list;
using std::tuple;
using std::deque;
using std::string;
using std::vector;
using std::ofstream;
using std::unordered_map;
using std::unordered_set;



namespace Simulator {

/**
 * @brief Structure that tries to simulate a CPU page table. The table is organized as follows:
 *        VAddr -> PAddr | Location | O_m
 *        VAddr: Page virtual address
 *        PAddr: Page physical address
 *        Location: Location of the page, can be in one of the following:
 *            { NOT_PRESENT, IN_SSD, IN_CPU, IN_GPU }
 *        O_m (in_transfer): mark if the page is currently in transfer
 * @note CPU page table size is dynamically increased when more pages are moved into the host
 *       memory, the current total pages will be printed during the reporting
 * @todo change VPN to Vaddr, PPN to PAddr
 */
class CPUPageTable {
  public:
    // CPU page table entries
    struct CPUPageTableEntry {
      Addr ppn;
      TensorLocation location;
      bool in_transfer;
    };
    CPUPageTable(size_t expected_size, ssize_t memory_line);

    /**
     * @brief create a CPU PTE using specified VAddr
     *
     * @param vpn VAddr of the page
     * @return pointer to created CPU PTE
     */
    CPUPageTableEntry* createEntry(Addr vpn);

    /**
     * @brief get a CPU PTE using specified VAddr
     *
     * @param vpn VAddr of the page
     * @return pointer to corresponding CPU PTE, null if VAddr is not mapped
     */
    CPUPageTableEntry* getEntry(Addr vpn);

    /**
     * @brief return if there is a CPU PTE that have VAddr specified
     * @param vpn VAddr of the page
     * @return whether there is a corresponding entry
     */
    bool exist(Addr vpn);

    /**
     * @brief allocate a CPU PTE for page that is stored in host memory
     * @param vpn VAddr of the page
     */
    void allocPTE(Addr vpn);

    /**
     * @brief mark the page as transferring (IN_TRANSFER)
     * @param vpn VAddr of the page
     */
    void markInTransferPTE(Addr vpn);

    /**
     * @brief mark the page as arrived (!IN_TRANSFER)
     * @param vpn VAddr of the page
     */
    void markArrivedPTE(Addr vpn);

    void AddInTransferPages(vector<Tensor *> &required_tensors);
    void AddInTransferPages(Addr start_addr);
    void RemoveInTransferPage(Addr start_addr);
    void RemoveAllInTransferPages();

    /**
     * @brief test if currently still have in transfer pages
     * @return if there is still in transfer pages
     */
    bool haveInTransferPages();
    size_t numInTransferPages();

    /**
     * @brief erase CPU PTE for page that is stored in host memory
     * @param vpn VAddr of the page
     */
    void erasePTE(Addr vpn);

    pair<size_t, size_t> getCapacity();

    long getMemoryLinePages();

    bool reachMemoryLine();

    // report the current status of the CPU PT
    void report();
    unordered_set<Addr> phys_page_avail;

  private:
    unordered_map<Addr, CPUPageTableEntry> page_table;

    unordered_set<Addr> in_transfer_pages;

    bool has_memory_line;
    unsigned long memory_line_pages;
    unsigned long total_memory_pages;
};

class GPUPageTable {
  public:
    enum EvcPolicy { RANDOM, LRU, GUIDED, DEEPUM, HOTNESS, HEURISTIC };
    class GPUPageTableEntry {
      public:
        Addr ppn;
        bool alloced_no_arrival;
      private:
        Tensor *tensor;
    };

    // eviction guide
    struct EvictCandidate {
        Addr vpn;
        Tensor *tensor;
        Eviction_P hotness;
        double exact_hotness;
    };

    struct EvictCandidateComp {
      bool operator()(const EvictCandidate &lhs, const EvictCandidate &rhs) const {
        return lhs.hotness < rhs.hotness;
      }
    };

    struct EvictCandidatePerfectComp {
      bool operator()(const EvictCandidate &lhs, const EvictCandidate &rhs) const {
        if (lhs.hotness == Eviction_P::Dead) return 0;
        if (rhs.hotness == Eviction_P::Dead) return 1;
        return lhs.exact_hotness > rhs.exact_hotness;
      }
    };

    GPUPageTable(unsigned long total_memory_pages, EvcPolicy policy, int candidate_cnt);

    GPUPageTableEntry* getEntry(Addr vpn);

    bool exist(Addr vpn);
    bool allocPTE(Addr vpn);
    void markArrivedPTE(Addr vpn);
    // erase the PTE for entry stored in host memory
    void erasePTE(Addr vpn);

    bool isFull();

    int hotness_candidates_last_kernel;
    priority_queue<EvictCandidate, vector<EvictCandidate>, EvictCandidateComp> hotness_candidates;
    tuple<Addr, GPUPageTableEntry, TensorLocation, EvictCandidate>
        selectEvictPTE(int kernel_id, bool is_pf);

    pair<size_t, size_t> getCapacity();
    void report();
    string reportLRUTable(int kernel_id);
    Tensor *searchTensorForPage(Addr vpn);

    void LRUPin(Addr addr);
    void LRUUnpin(Addr addr);

    friend class System;
    unordered_map<Addr, GPUPageTableEntry> page_table; // vpn -> entry
    std::set<int> tensors_in_pt;

  private:
    GPUPageTable();
    void LRUAccess(Addr addr);
    void LRURemove(Addr addr);
    Addr LRUGetLeastUsed();
    size_t LRUGetLeastUsed(vector<Addr>& lrus, size_t size);

    set<Addr> phys_page_avail;    // ppn
    unordered_set<Addr> alloced_no_arrival; // vpn

    // for eviction guide
    map<Addr, Tensor*> range_remap;
    // LRU
    list<Addr> lru_addrs;
    unordered_map<Addr, list<Addr>::iterator> lru_table;

    const unsigned long total_memory_pages;
    const EvcPolicy policy;
    const unsigned candidate_cnt;
};

//Input:
// t_00: Fully PF execution time from profiling (ms)
// t_10: (InputPF) Output Fully PF time from profiling (ms)
// t_11: no PF execution time from profiling (ms)
// r_input: the ratio of PF data in the input tensors (and not in-transfer)
// r_output: the ratio of PF data in the output tensors (and not in-transfer)
// r_input_ssd: the ratio of SSD PFs in the PFs (not in-transfer) in the input tensors
// r_output_ssd: the ratio of SSD PFs in the PFs (not in-transfer) in the output tensors
// s_input: total input tensor size (byte)
// s_output: total output tensor size (byte)
// BW_pcie: PCIe bandwidth (B/ms)
// BW_ssd: SSD bandwidth (B/ms)
// l_sys: System (CPU far-fault handling) latency (us)
// l_ssd: SSD latency (us)

//Output:
// delteT_PF: delta t for page fault handling (ms)
// BW_ssd_rest: The rest of SSD bandwidth useable when handling PF (for prefetching) (B/ms). Can be 0!!
// BW_pcie_rest: The rest of PCIe bandwidth usable when handling PF (for prefetching) （B/ms. Can be 0!!

void performance_model(double t_00, double t_10, double t_11, double r_input, double r_output,
                       double r_input_ssd, double r_output_ssd, long s_input, long s_output,
                       double BW_pcie, double BW_ssd, int l_sys, int l_ssd,
                       double& deltaT_PF, double& BW_ssd_rest, double& BW_pcie_rest);


class PageFaultInfo {
  public:
    PageFaultInfo() :
        not_presented_input_pages(0), not_presented_output_pages(0),
        CPU_to_GPU_faulted_input_pages(0), SSD_to_GPU_faulted_input_pages(0),
        CPU_to_GPU_faulted_output_pages(0), SSD_to_GPU_faulted_output_pages(0),
        total_input_pages(0), total_output_pages(0),
        in_transfer_pages(0), kernel(nullptr) {}
    PageFaultInfo(const CUDAKernel *kernel) :
        not_presented_input_pages(0), not_presented_output_pages(0),
        CPU_to_GPU_faulted_input_pages(0), SSD_to_GPU_faulted_input_pages(0),
        CPU_to_GPU_faulted_output_pages(0), SSD_to_GPU_faulted_output_pages(0),
        total_input_pages(0), total_output_pages(0),
        in_transfer_pages(0), kernel(kernel) {}
    PageFaultInfo &operator+=(const PageFaultInfo &rhs) {
      not_presented_input_pages += rhs.not_presented_input_pages;
      not_presented_output_pages += rhs.not_presented_output_pages;
      CPU_to_GPU_faulted_input_pages += rhs.CPU_to_GPU_faulted_input_pages;
      SSD_to_GPU_faulted_input_pages += rhs.SSD_to_GPU_faulted_input_pages;
      CPU_to_GPU_faulted_output_pages += rhs.CPU_to_GPU_faulted_output_pages;
      SSD_to_GPU_faulted_output_pages += rhs.SSD_to_GPU_faulted_output_pages;
      total_input_pages += rhs.total_input_pages;
      total_output_pages += rhs.total_output_pages;
      in_transfer_pages += rhs.in_transfer_pages;
      return *this;
    }
    unsigned long not_presented_input_pages;
    unsigned long not_presented_output_pages;
    unsigned long CPU_to_GPU_faulted_input_pages;
    unsigned long SSD_to_GPU_faulted_input_pages;
    unsigned long CPU_to_GPU_faulted_output_pages;
    unsigned long SSD_to_GPU_faulted_output_pages;
    unsigned long total_input_pages;
    unsigned long total_output_pages;
    unsigned long in_transfer_pages;
    const CUDAKernel *kernel;
};

class KernelRescheduleInfo {
  public:
    KernelRescheduleInfo(unsigned long first_scheduled_time,
                         unsigned long page_faulted_time) :
        first_scheduled_time(first_scheduled_time),
        page_faulted_time(page_faulted_time),
        kernel_started(false) {};
    const unsigned long first_scheduled_time;
    const unsigned long page_faulted_time;
    bool kernel_started;
    vector<PageFaultInfo> PF_info;
};

class System {
  public:
    System();
    ~System();
    // parameters
    // Frequency of GPU in Hz
    const unsigned long GPU_frequency_Hz;
    // Total memory size of GPU, in pages
    const unsigned long GPU_total_memory_pages;
    // PCIe latency in cycles
    const unsigned PCIe_latency_cycles;
    // PCIe bandwidth in byte per cycle
    const double CPU_PCIe_bandwidth_Bpc;
    const double GPU_PCIe_bandwidth_Bpc;
    const double SSD_PCIe_bandwidth_Bpc;
    // PCIe batch initiation interval
    const unsigned PCIe_batch_ii_cycle;
    // time taken for GPU to malloc a page in cycles
    const unsigned GPU_malloc_cycle_per_page;
    // time taken for GPU to free a page in cycles
    const unsigned GPU_free_cycle_per_page;
    // SSD read/write latency in cycles
    const unsigned SSD_read_latency_cycle;
    const unsigned SSD_write_latency_cycle;

    const bool should_use_movement_hints;
    // Migration policy
    const MigPolicy mig_policy;
    // GPU page table eviction policy
    const GPUPageTable::EvcPolicy evc_policy;
    const unsigned sys_prefetch_degree;
    const unsigned sys_num_candidate;
    // number of entry that should be batched in PCIe for each transfer
    const unsigned CPU_PCIe_batch_num;
    const unsigned GPU_PCIe_batch_num;
    const unsigned SSD_PCIe_batch_num;
    // number of alloc event that can done for each PCIe batching interval
    const unsigned alloc_batch_num;
    const double system_latency;
    const double SSD_latency;

    struct TensorHeuristicComp {
      bool operator()(const Tensor* lhs, const Tensor* rhs) const {
        if (lhs->hotness == Eviction_P::Dead ) return 0;
        else if (rhs->hotness == Eviction_P::Dead) return 1;
        return lhs->heuristic < rhs->heuristic;
      }
    };

    struct TensorHotnessComp {
      bool operator()(const Tensor* lhs, const Tensor* rhs) const {
        if (lhs->hotness == rhs->hotness) return lhs->addrs_in_GPU.size() > rhs->addrs_in_GPU.size();
        return lhs->hotness < rhs->hotness;
      }
    };

    priority_queue<Tensor*, vector<Tensor*>, TensorHotnessComp> tensor_evict_hotness_pq;
    priority_queue<Tensor*, vector<Tensor*>, TensorHeuristicComp> tensor_evict_heuristic_pq;
    // hardware components
    // GPU MMU
    // GPUMMU GPU_MMU;
    // CPU page table
    CPUPageTable CPU_PT;
    // GPU page table
    GPUPageTable GPU_PT;

    // prefetch handling
    deque<Addr> prefetch_SSD_queue;
    deque<Addr> prefetch_CPU_queue;
    // page fault handling
    deque<Addr> pf_CPU_queue;
    deque<Addr> pf_SSD_queue;
    deque<Addr> pf_alloc_queue;
    // preallocate handling
    deque<Addr> prealloc_queue;
    // eviction handling
    deque<Addr> preevict_SSD_queue;
    deque<Addr> preevict_CPU_queue;

    // other functions
    /**
     * @brief get the current cuda kernel
     * @return pointer to current executing cuda kernel
     */
    CUDAKernel* getCurrentKernel();

    /**
     * @brief get the next cuda kernel
     * @return pointer to next cuda kernel to be executed
     */
    CUDAKernel* getNextKernel();

    /**
     * @brief change current executing cuda kernel to the next in the list
     */
    void advanceCurrentKernel();

    int getMaxIteration();

    /**
     * @brief get the current iteration number.
     *
     * @return int current iteration number
     */
    int getCurrentIteration();

    void getCurrentMovementHints(vector<TensorMovementHint> &hints);

    /**
     * @brief get current total page fault number
     *
     * @return size_t total pf number
     */
    size_t getCurrentTotalPF();

    void generateRequiredTensorsForCurrentKernel();
    bool pageIsRequired(Addr start_address);
    bool tensorIsRequired(Tensor *tensor);

    /**
     * @brief DeepUM specific functions, adding the tensors of the target kernel into a set
     *        that contains tensors of several kernels in a running window that starts with
     *        the current executing kernel and spans for prefetch_degree
     *
     * @param kernel_num target kernel number where tensors of that kernel is added to the set
     */
    void addKernelTensorsToRunningWindow(int kernel_num);

    /**
     * @brief DeepUM specific functions, clear the running window
     */
    void clearRunningWindow();
    /**
     * @brief test if the target address is in the tensors contained in the running-window
     *
     * @param start_address address of the target page to be queried
     * @return if the target address in the running-window
     */
    bool pageInRunningWindowTensors(Addr start_address);

    /**
     * @brief initialize the LRU base iterators at the start of the kernel, ALWAYS call the function
     *        after possible addKernelTensorsToRunningWindow and clearRunningWindow
     */
    void deepUMSuggestInitialLRUBase();

    void LRUSuggestInitialLRUBase();

    /**
     * @brief update LRU base iterators
     *
     * @param suggested_lru_iter
     */
    void storeSuggestedLRUBase(list<Addr>::iterator suggested_lru_iter);

    /**
     * @brief get the stored LRU base iterators
     */
    list<Addr>::iterator getSuggestedLRUBase();

    KernelRescheduleInfo *reschedule_info;


    int batcher_evt_print_current = 0;
    const int batcher_evt_print_max = 50000;

    bool data_transferring = false;
  private:
    vector<CUDAKernel>::iterator current_kernel_iterator;
    unordered_set<Tensor *> tensor_running_window;
    unordered_set<Tensor *> current_kernel_required_tensors;
    list<Addr>::iterator current_lru_iterator;
    int max_iteration;
    int current_iteration;
    int current_hint_index;
};

class Stat {
  public:
    enum StatFileType {
      KernelStat,
      PCIeStat,
      EvcStat,
      TensorStat,
      FinalStat,
      LRUTableStat
    };

    Stat(string basename);
    ~Stat();

    void addKernelStat(int current_iter,
                       unsigned long start_time,
                       unsigned long end_time,
                       size_t CPU_used_pages,
                       size_t GPU_used_pages,
                       const CUDAKernel *kernel);
    void addKernelStat(int current_iter,
                       unsigned long start_time,
                       unsigned long end_time,
                       size_t CPU_used_pages,
                       size_t GPU_used_pages,
                       const CUDAKernel *kernel,
                       vector<PageFaultInfo> &PF_info);
    void addPCIeBWStat(int current_iter,
                       unsigned long start_time,
                       size_t incoming_pg_num,
                       size_t incoming_pg_SSD,
                       size_t incoming_pg_CPU,
                       size_t outgoing_pg_num,
                       size_t outgoing_pg_SSD,
                       size_t outgoing_pg_CPU,
                       size_t alloc_page_num);
    void addLRUTableStat(int current_iter,
                         const CUDAKernel *kernel,
                         string &LRU_table_report);
    void addEvcSelection(int current_iter,
                         unsigned long start_time,
                         int kernel_id,
                         TensorLocation to,
                         GPUPageTable::EvictCandidate& candidate);
    void addPFTensor(int current_iter,
                     Tensor *tensor,
                     int pg_total,
                     int in_transfer_cpu,
                     int in_transfer_ssd,
                     int in_transfer_unalloc,
                     int pf_cpu,
                     int pf_ssd,
                     int pf_unalloc);
    void addSizeInfo(long raw, long aligned);
    void printSizeInfo();

    void prepareOutputFiles(bool final_only=false);
    bool outputFileExists();
    void analyzeStat();
  private:
    void analyzeKernelStat();
    void analyzePCIeStat();
    void analyzeEvcStat();

    int getAllNumbersInLine(const string& input, vector<string>& output) const;
    void warn_corrupt_stat_file(const string &file) const;

    string output_file_basename;
    // <postfix, output file stream, if output initialization required>
    unordered_map<StatFileType, tuple<string, ofstream, bool>> output_files;

    long long raw_bytes = 0;
    long long aligned_bytes = 0;
};

} // namespace Simulator

#endif
