#include <iostream>
#include <queue>
#include <regex>
#include <math.h>
#include <vector>
#include <fstream>
#include <algorithm>

#include <cstdlib>
#include <exception>

#include "simulationComponents.h"
#include "printUtils.h"

using std::get;
using std::pow;
using std::pair;
using std::stoi;
using std::stol;
using std::stod;
using std::stold;
using std::regex;
using std::string;
using std::smatch;
using std::vector;
using std::getline;
using std::ifstream;
using std::ofstream;
using std::make_pair;
using std::make_tuple;
using std::regex_search;
using std::stringstream;
using std::priority_queue;
using Simulator::GPUPageTable;
using Simulator::TensorLocation;
using Simulator::TensorMovementHint;
using Simulator::MigPolicy;

// statistics and parameters to be filled by main function BEGIN
// CPU statistics
double CPU_PCIe_bandwidth_GBps = -1;
// GPU statistics
double GPU_frequency_GHz = -1;
double GPU_memory_size_GB = -1;
double GPU_PCIe_bandwidth_GBps = -1;
double GPU_malloc_uspB = -1;
// SSD statistics
double SSD_PCIe_bandwidth_GBps = -1;
double SSD_read_latency_us = -1;
double SSD_write_latency_us = -1;
// PCIe statistics
int PCIe_batch_size_in_page = -1;
// Other parameters
bool use_movement_hints = false;
string migration_policy_str;
string eviction_policy_str;
MigPolicy migration_policy;
GPUPageTable::EvcPolicy eviction_policy;
int prefetch_degree = -1;
int num_candidate = -1;
extern bool is_UVM;
extern int num_iteration;
// statistics and parameters END

extern vector<Tensor*> tensor_list;
extern vector<CUDAKernel> kernel_list;

extern vector<TensorMovementHint> movement_hints;
extern vector<EvictionGuideEntry> EvictionGuideTable;

extern long long memory_offset_intermediate;
extern long long memory_offset_weights;

extern double CPU_memory_line_GB;
extern double system_latency_us;
extern double SSD_latency_us;

namespace Simulator {

extern System *sim_sys;
extern Stat *sim_stat;

// CPUPageTable BEGIN ========================
CPUPageTable::CPUPageTable(size_t expected_size, ssize_t memory_line) {
  page_table.max_load_factor(0.7);
  page_table.reserve(expected_size);
  memory_line_pages = memory_line < 0 ? -1 : memory_line;
  has_memory_line = memory_line > 0;
  total_memory_pages = 0;
}

CPUPageTable::CPUPageTableEntry *CPUPageTable::createEntry(Addr vpn) {
  assert(!exist(vpn));
  page_table[vpn] = CPUPageTable::CPUPageTableEntry();
  page_table[vpn].in_transfer = false;
  page_table[vpn].location = TensorLocation::NOT_PRESENT;
  return &page_table[vpn];
}

CPUPageTable::CPUPageTableEntry *CPUPageTable::getEntry(Addr vpn) {
  if (!exist(vpn))
    return nullptr;
  return &page_table[vpn];
}

bool CPUPageTable::exist(Addr vpn) {
  return page_table.find(vpn) != page_table.end();
}

void CPUPageTable::allocPTE(Addr vpn) {
  assert(page_table.find(vpn) != page_table.end());
  if (phys_page_avail.size() == 0) {
    phys_page_avail.insert(total_memory_pages * PAGE_SIZE);
    total_memory_pages++;
    assert(has_memory_line || total_memory_pages <= memory_line_pages);
  }
  Addr ppn = *phys_page_avail.begin();
  phys_page_avail.erase(ppn);
  CPUPageTableEntry &entry = page_table[vpn];
  entry.ppn = ppn;
  entry.location = IN_CPU;
  markInTransferPTE(vpn);
}

void CPUPageTable::markInTransferPTE(Addr vpn) {
  assert(page_table.find(vpn) != page_table.end());
  CPUPageTableEntry &entry = page_table[vpn];
  entry.in_transfer = true;
}

void CPUPageTable::markArrivedPTE(Addr vpn) {
  assert(page_table.find(vpn) != page_table.end());
  CPUPageTableEntry &entry = page_table[vpn];
  entry.in_transfer = false;
}

void CPUPageTable::AddInTransferPages(vector<Tensor *> &required_tensors) {
  for (Tensor *tensor : required_tensors) {
    Addr start_addr = tensor->getGlobalOffset();
    long size = (long) tensor->size_in_byte;
    long total_pages = ceil((double) size / PAGE_SIZE);
    for (long page_num = 0; page_num < total_pages; page_num++) {
      Addr addr = start_addr + PAGE_SIZE * page_num;
      if (sim_sys->CPU_PT.getEntry(addr)->in_transfer)
        AddInTransferPages(addr);
    }
  }
}

void CPUPageTable::AddInTransferPages(Addr start_addr) {
  assert(sim_sys->tensorIsRequired(sim_sys->GPU_PT.searchTensorForPage(start_addr)));
  in_transfer_pages.insert(start_addr);
}

void CPUPageTable::RemoveInTransferPage(Addr start_addr) {
  if (in_transfer_pages.find(start_addr) != in_transfer_pages.end())
    in_transfer_pages.erase(start_addr);
}

void CPUPageTable::RemoveAllInTransferPages() {
  in_transfer_pages.clear();
}

bool CPUPageTable::haveInTransferPages() {
  return in_transfer_pages.size() > 0;
}

size_t CPUPageTable::numInTransferPages() {
  return in_transfer_pages.size();
}

void CPUPageTable::erasePTE(Addr vpn) {
  assert(page_table.find(vpn) != page_table.end());
  CPUPageTableEntry &entry = page_table[vpn];
  assert(phys_page_avail.find(entry.ppn) == phys_page_avail.end());
  assert(entry.location == TensorLocation::IN_CPU);
  entry.location = TensorLocation::NOT_PRESENT;
  entry.in_transfer = false;
  phys_page_avail.insert(entry.ppn);
}

pair<size_t, size_t> CPUPageTable::getCapacity() {
  return make_pair(total_memory_pages - phys_page_avail.size(), total_memory_pages);
}

long CPUPageTable::getMemoryLinePages() {
  if (has_memory_line)
    return total_memory_pages;
  return -1;
}

bool CPUPageTable::reachMemoryLine() {
  assert(has_memory_line || total_memory_pages <= memory_line_pages);
  return !has_memory_line && total_memory_pages - phys_page_avail.size() == memory_line_pages;
}

void CPUPageTable::report() {
  printf("  CPU PT Available/CurrentTotal/MemoryLine=<%ld/%ld/%ld>\n",
      phys_page_avail.size(), total_memory_pages, memory_line_pages);
}
// CPUPageTable END ========================

// GPUPageTable BEGIN ========================
GPUPageTable::GPUPageTable(unsigned long total_memory_pages, EvcPolicy policy, int candidate_cnt) :
    total_memory_pages(total_memory_pages), policy(policy), candidate_cnt(candidate_cnt) {
  // phys_page_avail.reserve(total_memory_pages);
  page_table.max_load_factor(0.8);
  page_table.reserve(total_memory_pages);
  for (unsigned long page_num = 0; page_num < total_memory_pages; page_num++) {
    phys_page_avail.insert(page_num * PAGE_SIZE);
  }
  // initialize eviction guide specific data structures
  for (Tensor *tensor : tensor_list) {
    assert(range_remap.find(tensor->getGlobalOffset()) == range_remap.end());
    range_remap[tensor->getGlobalOffset()] = tensor;
  }
}

GPUPageTable::GPUPageTableEntry *GPUPageTable::getEntry(Addr vpn) {
  if (!exist(vpn))
    return nullptr;
  LRUAccess(vpn);
  return &page_table.at(vpn);
}

bool GPUPageTable::exist(Addr vpn) {
  return page_table.find(vpn) != page_table.end();
}

bool GPUPageTable::allocPTE(Addr vpn) {
  if (page_table.find(vpn) != page_table.end()){
    searchTensorForPage(vpn)->access_count++;
    return true;
  }
  
  if (phys_page_avail.size() == 0)
    return false;
  Addr ppn = *phys_page_avail.begin();
  phys_page_avail.erase(phys_page_avail.begin());
  GPUPageTableEntry &entry = page_table[vpn];
  entry.ppn = ppn;
  entry.alloced_no_arrival = true;
  searchTensorForPage(vpn)->access_count++;
  searchTensorForPage(vpn)->addrs_in_GPU.insert(vpn);
  assert(phys_page_avail.size() + page_table.size() == total_memory_pages);
  LRUAccess(vpn);
  return true;
}

void GPUPageTable::markArrivedPTE(Addr vpn) {
  assert(page_table.find(vpn) != page_table.end());
  GPUPageTableEntry &entry = page_table[vpn];
  entry.alloced_no_arrival = false;
}

void GPUPageTable::erasePTE(Addr vpn) {
  if (page_table.find(vpn) == page_table.end())
    return;
  assert(page_table[vpn].alloced_no_arrival == false);
  Addr ppn = page_table[vpn].ppn;
  searchTensorForPage(vpn)->addrs_in_GPU.erase(vpn);
  page_table.erase(vpn);
  phys_page_avail.insert(ppn);
  assert(phys_page_avail.size() + page_table.size() == total_memory_pages);
  LRURemove(vpn);
}

bool GPUPageTable::isFull() {
  return phys_page_avail.size() == 0;
}

/**
 * @brief Select the 
 *
 * @param kernel_id the kernel id when the eviction is ordered
 * @param is_pf 
 * @return 
 */
tuple<Addr, GPUPageTable::GPUPageTableEntry, TensorLocation, GPUPageTable::EvictCandidate>
    GPUPageTable::selectEvictPTE(int kernel_id, bool is_pf) {
  // GPU memory is full, eviction required
  tuple<Addr, GPUPageTable::GPUPageTableEntry, TensorLocation, EvictCandidate> evicted_entry;
  switch (policy) {
    case EvcPolicy::HEURISTIC: {
      while (sim_sys->tensor_evict_heuristic_pq.top()->addrs_in_GPU.empty())
      {
        std::cout << "All addrs of tensor" << sim_sys->tensor_evict_heuristic_pq.top()->tensor_id << " is evicted" << std::endl;
        sim_sys->tensor_evict_heuristic_pq.pop();
      }
      Tensor* candidate_tensor = sim_sys->tensor_evict_heuristic_pq.top();
      
      EvictCandidate &ret_candidate = get<3>(evicted_entry);
      ret_candidate.vpn = *candidate_tensor->addrs_in_GPU.begin();
      candidate_tensor->addrs_in_GPU.erase(ret_candidate.vpn);
      ret_candidate.tensor = searchTensorForPage(ret_candidate.vpn);
      ret_candidate.hotness = Eviction_P::Invalid;
      ret_candidate.exact_hotness = Eviction_P::Invalid;
      
      get<0>(evicted_entry) = ret_candidate.vpn;
      get<1>(evicted_entry) = page_table[ret_candidate.vpn];
      if (candidate_tensor->hotness == Eviction_P::Dead) {
        get<2>(evicted_entry) = NOT_PRESENT;
      } else if (candidate_tensor->addrs_in_GPU.size() > sim_sys->CPU_PT.phys_page_avail.size()/5 ) {
        get<2>(evicted_entry) = IN_SSD;
      } else {
        get<2>(evicted_entry) = IN_CPU;
      }
      break;
    }
    case EvcPolicy::HOTNESS: {
      while (sim_sys->tensor_evict_hotness_pq.top()->addrs_in_GPU.empty())
      {
        std::cout << "All addrs of tensor" << sim_sys->tensor_evict_hotness_pq.top()->tensor_id << " is evicted" << std::endl;
        sim_sys->tensor_evict_hotness_pq.pop();
      }
      Tensor* candidate_tensor = sim_sys->tensor_evict_hotness_pq.top();
      
      EvictCandidate &ret_candidate = get<3>(evicted_entry);
      ret_candidate.vpn = *candidate_tensor->addrs_in_GPU.begin();
      candidate_tensor->addrs_in_GPU.erase(ret_candidate.vpn);
      ret_candidate.tensor = searchTensorForPage(ret_candidate.vpn);
      ret_candidate.hotness = candidate_tensor->hotness;
      ret_candidate.exact_hotness = Eviction_P::Invalid;
      //if (kernel_id == 36) std::cout << "Ret Tensor" << ret_candidate.tensor->tensor_id << " addr " << ret_candidate.vpn << std::endl;
      get<0>(evicted_entry) = ret_candidate.vpn;
      get<1>(evicted_entry) = page_table[ret_candidate.vpn];
      if (ret_candidate.hotness == Dead) {
        get<2>(evicted_entry) = NOT_PRESENT;
      } else if (ret_candidate.hotness == Eviction_P::Hot) {
        get<2>(evicted_entry) = IN_SSD;
      } else if (ret_candidate.hotness == Eviction_P::Medium) {
        get<2>(evicted_entry) = IN_CPU;
      } else if (ret_candidate.hotness == Eviction_P::Cold) {
        get<2>(evicted_entry) = IN_CPU;
      }
      break;
    }
    case EvcPolicy::RANDOM: {
      // select random entry
      int bucket, bucket_size;
      unordered_map<Addr, GPUPageTable::GPUPageTableEntry>::local_iterator rand_it;
      do {
        do { // magic that randomly select from unordered map in const time
            bucket = rand() % page_table.bucket_count();
            bucket_size = page_table.bucket_size(bucket);
        } while (bucket_size == 0);
        rand_it = std::next(page_table.begin(bucket), rand() % bucket_size);
      } while (sim_sys->CPU_PT.getEntry(rand_it->first)->in_transfer);

      EvictCandidate &ret_candidate = get<3>(evicted_entry);
      ret_candidate.vpn = rand_it->first;
      ret_candidate.tensor = searchTensorForPage(ret_candidate.vpn);
      ret_candidate.hotness = Eviction_P::Invalid;
      ret_candidate.exact_hotness = Eviction_P::Invalid;

      get<0>(evicted_entry) = rand_it->first;
      get<1>(evicted_entry) = rand_it->second;
      // get<2>(evicted_entry) = (rand() & 1) ? IN_SSD : IN_CPU;
      get<2>(evicted_entry) = IN_CPU;
      break;
    }
    case EvcPolicy::LRU: {
      sim_sys->LRUSuggestInitialLRUBase();
      auto lru_it = sim_sys->getSuggestedLRUBase();
      assert(lru_it != lru_addrs.end());

      EvictCandidate &ret_candidate = get<3>(evicted_entry);
      ret_candidate.vpn = *lru_it;
      ret_candidate.tensor = searchTensorForPage(ret_candidate.vpn);
      ret_candidate.hotness = Eviction_P::Invalid;
      ret_candidate.exact_hotness = Eviction_P::Invalid;

      get<0>(evicted_entry) = ret_candidate.vpn;
      get<1>(evicted_entry) = page_table.at(ret_candidate.vpn);
      // get<2>(evicted_entry) = (rand() & 1) ? IN_SSD : IN_CPU;
      get<2>(evicted_entry) = IN_CPU;
      break;
    }
    case EvcPolicy::GUIDED: {
      priority_queue<EvictCandidate, vector<EvictCandidate>, EvictCandidateComp> candidates;
      for (int i = 0; i < candidate_cnt; i++) {
        // select random entry
        int bucket, bucket_size;
        unordered_map<Addr, GPUPageTable::GPUPageTableEntry>::local_iterator rand_it;
        do {
          do { // magic that randomly select from unordered map in const time
              bucket = rand() % page_table.bucket_count();
              bucket_size = page_table.bucket_size(bucket);
          } while (bucket_size == 0);
          rand_it = std::next(page_table.begin(bucket), rand() % bucket_size);
        } while (sim_sys->CPU_PT.getEntry(rand_it->first)->in_transfer);
        // generate and add candidate to priority queue
        EvictCandidate candidate;
        candidate.vpn = rand_it->first;
        candidate.tensor = searchTensorForPage(candidate.vpn);
        candidate.hotness = Eviction_P::Invalid;
        candidate.exact_hotness = Eviction_P::Invalid;
        candidates.emplace(candidate);
      }
      const EvictCandidate& target_candidate = candidates.top();
      EvictCandidate &ret_candidate = get<3>(evicted_entry);
      ret_candidate.vpn = target_candidate.vpn;
      ret_candidate.tensor = target_candidate.tensor;
      ret_candidate.hotness = target_candidate.hotness;
      ret_candidate.exact_hotness = target_candidate.exact_hotness;

      get<0>(evicted_entry) = target_candidate.vpn;
      get<1>(evicted_entry) = page_table[target_candidate.vpn];
      if (target_candidate.hotness == Dead) {
        get<2>(evicted_entry) = NOT_PRESENT;
      } else if (target_candidate.hotness == Eviction_P::Hot) {
        get<2>(evicted_entry) = IN_SSD;
      } else if (target_candidate.hotness == Eviction_P::Medium) {
        get<2>(evicted_entry) = IN_CPU;
      } else if (target_candidate.hotness == Eviction_P::Cold) {
        get<2>(evicted_entry) = IN_CPU;
      }
      break;
    }
    case EvcPolicy::DEEPUM: {
      EvictCandidate &ret_candidate = get<3>(evicted_entry);

      if (!is_pf) {
        sim_sys->LRUSuggestInitialLRUBase();
        auto lru_base_it = sim_sys->getSuggestedLRUBase();
        // not page faulted, skip LRUs that resident in running window
        while (lru_base_it != lru_addrs.end() && sim_sys->pageInRunningWindowTensors(*lru_base_it)) {
          lru_base_it++;
        }
        if (lru_base_it == lru_addrs.end()) {
          ret_candidate.vpn = 0;
          ret_candidate.tensor = nullptr;
          // bypass validation check after break, this indicate such candidate is not found
          sim_sys->storeSuggestedLRUBase(lru_addrs.end());
          return evicted_entry;
        }
        ret_candidate.vpn = *lru_base_it;
        sim_sys->storeSuggestedLRUBase(++lru_base_it);
      } else {
        auto lru_it = lru_addrs.begin();
        auto lru_base_it = sim_sys->getSuggestedLRUBase();
        while (lru_it != lru_addrs.end() && sim_sys->CPU_PT.getEntry(*lru_it)->in_transfer) {
          lru_it++;
        }
        assert(lru_it != lru_addrs.end());
        if (lru_it == lru_base_it) {
          sim_sys->storeSuggestedLRUBase(++lru_base_it);
        }
        ret_candidate.vpn = *lru_it;
      }

      ret_candidate.tensor = searchTensorForPage(ret_candidate.vpn);
      ret_candidate.hotness = Eviction_P::Invalid;
      ret_candidate.exact_hotness = Eviction_P::Invalid;

      get<0>(evicted_entry) = ret_candidate.vpn;
      if (page_table.find(ret_candidate.vpn) == page_table.end()) {
        eprintf("VPN: %ld Tid: %d is_pf: %d\n",
            ret_candidate.vpn, ret_candidate.tensor->tensor_id, is_pf);
        assert(false);
      }
      get<1>(evicted_entry) = page_table.at(ret_candidate.vpn);
      if (ret_candidate.tensor->is_alive(kernel_id))
        get<2>(evicted_entry) = IN_CPU;
      else
        get<2>(evicted_entry) = NOT_PRESENT;
      break;
    }
    default:
      assert(false);
  }
  // sanity check
  assert(page_table.find(get<0>(evicted_entry)) != page_table.end());
  return evicted_entry;
}

pair<size_t, size_t> GPUPageTable::getCapacity() {
  return make_pair(page_table.size(), total_memory_pages);
}

void GPUPageTable::report() {
  printf("  GPU PT Available/Assigned/Total=<%ld/%ld/%ld>\n",
      phys_page_avail.size(), page_table.size(), total_memory_pages);
}

Tensor *GPUPageTable::searchTensorForPage(Addr vpn) {
  map<Addr, Tensor *>::iterator it = range_remap.upper_bound(vpn);
  assert(it != range_remap.begin());
  Tensor *tensor = (--it)->second;
  assert(vpn >= tensor->getGlobalOffset() &&
         vpn < tensor->getGlobalOffset() + tensor->size_in_byte);
  return tensor;
}

// TODO: change this
void GPUPageTable::LRUPin(Addr addr) {
  if (lru_table.find(addr) == lru_table.end())
    return;
  lru_addrs.erase(lru_table[addr]);
  lru_addrs.push_front(addr);
  lru_table[addr] = lru_addrs.begin();
  // sanity check
  assert(lru_table.size() == page_table.size());
}

// TODO: change this
void GPUPageTable::LRUUnpin(Addr addr) {
  if (lru_table.find(addr) == lru_table.end())
    return;
  lru_addrs.erase(lru_table[addr]);
  lru_addrs.push_back(addr);
  lru_table[addr] = --lru_addrs.end();
  // sanity check
  assert(lru_table.size() == page_table.size());
}

void GPUPageTable::LRUAccess(Addr addr) {
  assert(addr < memory_offset_intermediate + memory_offset_weights);
  auto lru_item = lru_table.find(addr);
  bool change_suggestion = false;
  if (lru_item != lru_table.end()) {
    if (lru_item->second == sim_sys->getSuggestedLRUBase())
      change_suggestion = true;
    lru_addrs.erase(lru_item->second);
  }
  lru_addrs.push_back(addr);
  lru_table[addr] = --lru_addrs.end();
  if (change_suggestion)
    sim_sys->storeSuggestedLRUBase(lru_addrs.begin());
  // sanity check
  assert(page_table.find(addr) != page_table.end());
  assert(lru_table.size() == page_table.size());
}

void GPUPageTable::LRURemove(Addr addr) {
  auto lru_item = lru_table.find(addr);
  bool change_suggestion = false;
  assert(lru_item != lru_table.end());
  if (lru_item->second == sim_sys->getSuggestedLRUBase())
    change_suggestion = true;
  lru_addrs.erase(lru_item->second);
  lru_table.erase(addr);
  if (change_suggestion)
    sim_sys->storeSuggestedLRUBase(lru_addrs.begin());
  // sanity check
  assert(page_table.find(addr) == page_table.end());
  assert(lru_table.size() == page_table.size());
}

Addr GPUPageTable::LRUGetLeastUsed() {
  assert(lru_addrs.size() >= 1);
  return lru_addrs.front();
}

size_t GPUPageTable::LRUGetLeastUsed(vector<Addr>& lrus, size_t size) {
  assert(lru_addrs.size() >= 1);
  size_t actural_size = lru_addrs.size() > size ? size : lru_addrs.size();
  lrus.clear();
  auto it = lru_addrs.begin();
  for (int i = 0; i < actural_size; i++) {
    lrus.push_back(*it++);
  }
  return actural_size;
}

string GPUPageTable::reportLRUTable(int kernel_id) {
  if (lru_addrs.size() == 0) return "Total LRU Table size: 0\n";
  string summary_out, exact_out;
  char buf[100];
  snprintf(buf, sizeof(buf), "Total LRU Table size: %ld\n", lru_addrs.size());
  summary_out += string(buf);

  Tensor *current_tensor = nullptr;
  int tensor_coalescing_cnt = 0;
  int hotness_coalescing_cnt = 0;
  for (auto it = lru_addrs.begin(); it != lru_addrs.end(); ++it) {
    Addr addr = *it;
    Tensor *tensor = searchTensorForPage(addr);
    if (!current_tensor || tensor->tensor_id == current_tensor->tensor_id) {
      current_tensor = tensor;
      tensor_coalescing_cnt++;
    } else {
      Eviction_P hotness = EvictionGuideTable[kernel_id].entry[current_tensor];
      double exact_hotness = hotness == Eviction_P::Dead ? -1 :
          EvictionGuideTable[kernel_id].absolute_time_entry[current_tensor];
      snprintf(buf, sizeof(buf), "Num_pages:%10d Tensor:%5d Hotness:%7s Exact:%f\n",
          tensor_coalescing_cnt, current_tensor->tensor_id, print_eviction_array[hotness].c_str(), exact_hotness);
      exact_out += string(buf);

      if (EvictionGuideTable[kernel_id].entry[tensor] == hotness) {
        hotness_coalescing_cnt += tensor_coalescing_cnt;
      } else {
        snprintf(buf, sizeof(buf), "Num_pages:%10d Hotness:%7s\n",
            hotness_coalescing_cnt + tensor_coalescing_cnt, print_eviction_array[hotness].c_str());
        summary_out += string(buf);
        hotness_coalescing_cnt = 0;
      }

      current_tensor = tensor;
      tensor_coalescing_cnt = 1;
    }
  }
  Eviction_P hotness = EvictionGuideTable[kernel_id].entry[current_tensor];
  double exact_hotness = hotness == Eviction_P::Dead ? -1 :
      EvictionGuideTable[kernel_id].absolute_time_entry[current_tensor];
  snprintf(buf, sizeof(buf), "Num_pages:%10d Tensor:%5d Hotness:%7s Exact:%f\n",
      tensor_coalescing_cnt, current_tensor->tensor_id, print_eviction_array[hotness].c_str(), exact_hotness);
  exact_out += string(buf);
  hotness_coalescing_cnt += tensor_coalescing_cnt;

  snprintf(buf, sizeof(buf), "Num_pages:%10d Hotness:%7s\n",
      hotness_coalescing_cnt, print_eviction_array[hotness].c_str());
  summary_out += string(buf);

  return "Tail (LRU)\n" + summary_out + "Head (MRU)\n" + "=====\n" +
         "Tail (LRU)\n" + exact_out + "Head (MRU)\n";
}
// GPUPageTable END ========================

// System BEGIN ========================
System::System() :
    GPU_frequency_Hz(GPU_frequency_GHz * pow(10, 9)),
    GPU_total_memory_pages(round(GPU_memory_size_GB / PAGE_SIZE * pow(1024, 3))),
    PCIe_latency_cycles(0), // parameter not used
    CPU_PCIe_bandwidth_Bpc(CPU_PCIe_bandwidth_GBps * pow(10, 9) / GPU_frequency_Hz),
    GPU_PCIe_bandwidth_Bpc(GPU_PCIe_bandwidth_GBps * pow(10, 9) / GPU_frequency_Hz),
    SSD_PCIe_bandwidth_Bpc(SSD_PCIe_bandwidth_GBps * pow(10, 9) / GPU_frequency_Hz),
    PCIe_batch_ii_cycle((PCIe_batch_size_in_page * PAGE_SIZE) / GPU_PCIe_bandwidth_Bpc),
    GPU_malloc_cycle_per_page(round(GPU_malloc_uspB * PAGE_SIZE / pow(10, 6) * GPU_frequency_Hz)),
    GPU_free_cycle_per_page(0), // parameter not used
    SSD_read_latency_cycle(SSD_read_latency_us / pow(10, 6) * GPU_frequency_Hz),
    SSD_write_latency_cycle(SSD_write_latency_us / pow(10, 6) * GPU_frequency_Hz),
    should_use_movement_hints(use_movement_hints),
    mig_policy(migration_policy),
    evc_policy(eviction_policy),
    sys_prefetch_degree(prefetch_degree),
    sys_num_candidate(num_candidate),
    CPU_PCIe_batch_num(PCIe_batch_size_in_page / GPU_PCIe_bandwidth_GBps * CPU_PCIe_bandwidth_GBps),
    GPU_PCIe_batch_num(PCIe_batch_size_in_page),
    SSD_PCIe_batch_num(PCIe_batch_size_in_page / GPU_PCIe_bandwidth_GBps * SSD_PCIe_bandwidth_GBps),
    alloc_batch_num(PCIe_batch_ii_cycle / GPU_malloc_cycle_per_page),
    system_latency(system_latency_us),
    SSD_latency(SSD_latency_us),
    CPU_PT(CPUPageTable(
        (memory_offset_intermediate + memory_offset_weights) / PAGE_SIZE,
        CPU_memory_line_GB < 0 ? -1 : CPU_memory_line_GB * pow(1024, 3) / 4096)),
    GPU_PT(GPUPageTable(GPU_total_memory_pages, evc_policy, num_candidate)) {
  printf("========== Simulation Setting ==========\n");
  printf(" General:\n");
  printf("  Using UVM:                  %s\n", is_UVM ? "True" : "False");
  printf("  Prefetch Enabled:           %s\n", use_movement_hints ? "True" : "False");
  printf("  Eviction Policy:            %s\n", eviction_policy_str.c_str());
  printf(" CPU:\n");
  printf("  CPU PCIe BW Byte/Cycle:     %-10f\n", CPU_PCIe_bandwidth_Bpc);
  printf("  CPU PCIe Batch Num:         %-10d\n", CPU_PCIe_batch_num);
  printf("  CPU PCIe BW Utilization:    %-6.3f\n",
      (double) CPU_PCIe_batch_num / PCIe_batch_ii_cycle / CPU_PCIe_bandwidth_Bpc * PAGE_SIZE * 100);
  printf("  CPU Memory Line:            %-10ld\n", CPU_PT.getMemoryLinePages());
  printf(" GPU:\n");
  printf("  GPU Frequency Hz:           %-10ld\n", GPU_frequency_Hz);
  printf("  GPU Total Memory Pages:     %-10ld\n", GPU_total_memory_pages);
  printf("  GPU PCIe BW Byte/Cycle:     %-10f\n", GPU_PCIe_bandwidth_Bpc);
  printf("  GPU PCIe Batch Num:         %-10d\n", GPU_PCIe_batch_num);
  printf("  GPU PCIe BW Utilization:    %-6.3f\n",
      (double) GPU_PCIe_batch_num / PCIe_batch_ii_cycle / GPU_PCIe_bandwidth_Bpc * PAGE_SIZE * 100);
  printf("  GPU Malloc Cycle/Page:      %-10d\n", GPU_malloc_cycle_per_page);
  printf("  GPU Free Cycle/Page:        %-10d\n", GPU_free_cycle_per_page);
  printf(" SSD:\n");
  printf("  SSD PCIe BW Byte/Cycle:     %-10f\n", SSD_PCIe_bandwidth_Bpc);
  printf("  SSD PCIe Batch Num:         %-10d\n", SSD_PCIe_batch_num);
  printf("  SSD PCIe BW Utilization:    %-6.3f\n",
      (double) SSD_PCIe_batch_num / PCIe_batch_ii_cycle / SSD_PCIe_bandwidth_Bpc * PAGE_SIZE * 100);
  printf("  SSD Read Latency Cycle:     %-10d\n", SSD_read_latency_cycle);
  printf("  SSD Write Latency Cycle:    %-10d\n", SSD_write_latency_cycle);
  printf(" PCIe:\n");
  printf("  PCIe_latency_cycles:        %-10d\n", PCIe_latency_cycles);
  printf("  PCIe Batch II Cycle:        %-10d\n", PCIe_batch_ii_cycle);
  printf("  PCIe Batch Size Page:       %-10d\n", PCIe_batch_size_in_page);
  printf("  PCIe Alloc Batch Size:      %-10d\n", alloc_batch_num);
  printf("======== Simulation Setting END ========\n");

  current_kernel_iterator = kernel_list.begin();
  reschedule_info = nullptr;
  max_iteration = num_iteration;
  current_iteration = 0;
  current_hint_index = 0;
}

System::~System() {
  delete reschedule_info;
}

CUDAKernel* System::getCurrentKernel() {
  return &(*current_kernel_iterator);
}

CUDAKernel* System::getNextKernel() {
  auto it = current_kernel_iterator + 1;
  if (it == kernel_list.end()) {
    return &(*kernel_list.begin());
  }
  return &(*it);
}

void System::advanceCurrentKernel() {
  current_kernel_iterator++;
  if (current_kernel_iterator == kernel_list.end()) {
    current_kernel_iterator = kernel_list.begin();
    current_iteration++;
  }
}

int System::getMaxIteration() {
  return max_iteration;
}

int System::getCurrentIteration() {
  return current_iteration;
}

void System::getCurrentMovementHints(vector<TensorMovementHint> &hints) {
  hints.clear();
  while (movement_hints.size() &&
      movement_hints[current_hint_index].issued_kernel_id % kernel_list.size() ==
      current_kernel_iterator->kernel_id) {
    hints.push_back(movement_hints[current_hint_index]);
    current_hint_index = (current_hint_index + 1) % movement_hints.size();
  }
}

size_t System::getCurrentTotalPF() {
  return pf_CPU_queue.size() + pf_SSD_queue.size() + pf_alloc_queue.size();
}

void System::generateRequiredTensorsForCurrentKernel() {
  current_kernel_required_tensors.clear();
  getCurrentKernel()->getRequiredTensors(current_kernel_required_tensors);
}

bool System::pageIsRequired(Addr start_address) {
  return tensorIsRequired(GPU_PT.searchTensorForPage(start_address));
}

bool System::tensorIsRequired(Tensor *tensor) {
  return current_kernel_required_tensors.find(tensor) != current_kernel_required_tensors.end();
}

void System::clearRunningWindow() {
  tensor_running_window.clear();
}

void System::addKernelTensorsToRunningWindow(int kernel_num) {
  kernel_list[kernel_num].getRequiredTensors(tensor_running_window);
}

bool System::pageInRunningWindowTensors(Addr start_address) {
  Tensor *tensor = sim_sys->GPU_PT.searchTensorForPage(start_address);
  return tensor_running_window.find(tensor) != tensor_running_window.end();
}

void System::deepUMSuggestInitialLRUBase() {
  current_lru_iterator = sim_sys->GPU_PT.lru_addrs.begin();
  while (current_lru_iterator != sim_sys->GPU_PT.lru_addrs.end() &&
         sim_sys->pageInRunningWindowTensors(*current_lru_iterator)) {
    current_lru_iterator++;
  }
}

void System::LRUSuggestInitialLRUBase() {
  current_lru_iterator = sim_sys->GPU_PT.lru_addrs.begin();
  while (current_lru_iterator != sim_sys->GPU_PT.lru_addrs.end() &&
         sim_sys->CPU_PT.getEntry(*current_lru_iterator)->in_transfer) {
    current_lru_iterator++;
  }
}

void System::storeSuggestedLRUBase(list<Addr>::iterator suggested_lru_iter) {
  current_lru_iterator = suggested_lru_iter;
}

list<Addr>::iterator System::getSuggestedLRUBase() {
  return current_lru_iterator;
}
// System END ========================

// Stat BEGIN ========================
Stat::Stat(string basename) : output_file_basename(basename) {
  output_file_basename = basename;
  // declare output files
  output_files.emplace(KernelStat, make_tuple("kernel", ofstream(), true));
  output_files.emplace(PCIeStat, make_tuple("pcie", ofstream(), true));
  output_files.emplace(EvcStat, make_tuple("evc", ofstream(), true));
  output_files.emplace(TensorStat, make_tuple("pf_tensor", ofstream(), true));
  output_files.emplace(FinalStat, make_tuple("final", ofstream(), false));
  output_files.emplace(LRUTableStat, make_tuple("lru", ofstream(), false));

  // process file names
  for (auto pair = output_files.begin(); pair != output_files.end(); pair++) {
    get<0>(pair->second) = output_file_basename + "." + get<0>(pair->second);
  }
}

Stat::~Stat() {
  for (auto pair = output_files.begin(); pair != output_files.end(); pair++) {
    ofstream& fout = get<1>(pair->second);
    if (fout.is_open()) {
      fout << -1;
      fout.close();
    }
  }
}

void Stat::addKernelStat(int current_iter,
                         unsigned long start_time,
                         unsigned long end_time,
                         size_t CPU_used_pages,
                         size_t GPU_used_pages,
                         const CUDAKernel *kernel) {
  vector<PageFaultInfo> info;
  addKernelStat(current_iter, start_time, end_time,
      CPU_used_pages, GPU_used_pages, kernel, info);
}

void Stat::addKernelStat(int current_iter,
                         unsigned long start_time,
                         unsigned long end_time,
                         size_t CPU_used_pages,
                         size_t GPU_used_pages,
                         const CUDAKernel *kernel,
                         vector<PageFaultInfo> &PF_info) {
  ofstream& fout = get<1>(output_files.at(KernelStat));

  unsigned in_transfer_pages = 0;
  unsigned PF_from_CPU = 0;
  unsigned PF_from_SSD = 0;
  unsigned PF_unalloc = 0;
  for (PageFaultInfo info : PF_info) {
    in_transfer_pages += info.in_transfer_pages;
    PF_from_CPU += info.CPU_to_GPU_faulted_input_pages + info.CPU_to_GPU_faulted_output_pages;
    PF_from_SSD += info.SSD_to_GPU_faulted_input_pages + info.SSD_to_GPU_faulted_output_pages;
    PF_unalloc += info.not_presented_input_pages + info.not_presented_output_pages;
  }

  fout << current_iter << "+" << kernel->kernel_id <<
      ":[" << start_time << "," << end_time << "]" <<
      "(" << kernel->execution_cycles << "," << kernel->pf_execution_cycles << ")" <<
      "<" << in_transfer_pages << "," << PF_from_CPU << "," << PF_from_SSD << "," << PF_unalloc << ">" <<
      "(" << CPU_used_pages << "," << GPU_used_pages << ")\n";
}

void Stat::addPCIeBWStat(int current_iter,
                         unsigned long start_time,
                         size_t incoming_pg_num,
                         size_t incoming_pg_SSD,
                         size_t incoming_pg_CPU,
                         size_t outgoing_pg_num,
                         size_t outgoing_pg_SSD,
                         size_t outgoing_pg_CPU,
                         size_t alloc_page_num) {
  ofstream& fout = get<1>(output_files.at(PCIeStat));

  if (incoming_pg_num == 0 && outgoing_pg_num == 0)
    return;

  assert(incoming_pg_SSD + incoming_pg_CPU == incoming_pg_num &&
         outgoing_pg_SSD + outgoing_pg_CPU == outgoing_pg_num);


  fout << current_iter << "[" << start_time << "]" << alloc_page_num <<
      "(" << incoming_pg_num << "=" << incoming_pg_SSD << "+" << incoming_pg_CPU << ")" <<
      "(" << outgoing_pg_num << "=" << outgoing_pg_SSD << "+" << outgoing_pg_CPU << ")\n";
}

void Stat::addEvcSelection(int current_iter,
                           unsigned long start_time,
                           int kernel_id,
                           TensorLocation to,
                           GPUPageTable::EvictCandidate& candidate) {
  ofstream& fout = get<1>(output_files.at(EvcStat));
  fout << current_iter << "+" << kernel_id <<
      "[" << start_time << "]" << to << "," << candidate.vpn << "," <<
      candidate.tensor->tensor_id << "," << candidate.hotness << "\n";
}

void Stat::addPFTensor(int current_iter,
                       Tensor *tensor,
                       int pg_total,
                       int in_transfer_cpu,
                       int in_transfer_ssd,
                       int in_transfer_unalloc,
                       int pf_cpu,
                       int pf_ssd,
                       int pf_unalloc) {
  ofstream& fout = get<1>(output_files.at(TensorStat));
  fout << current_iter << "[" << tensor->getGlobalOffset() << "," <<
      tensor->getGlobalOffset() + tensor->size_in_byte << "]" <<
      pg_total <<
      "(" << in_transfer_cpu << "," << in_transfer_ssd << "," << in_transfer_unalloc <<")(" <<
      pf_cpu << "," << pf_ssd << "," << pf_unalloc << ")\n";
}

void Stat::addLRUTableStat(int current_iter,
                           const CUDAKernel *kernel,
                           string &LRU_table_report) {
  string filename = get<0>(output_files.at(LRUTableStat)) + std::to_string(kernel->kernel_id);
  ofstream fout = ofstream(filename);
  fout << LRU_table_report;
  fout.close();
}

void Stat::prepareOutputFiles(bool final_only) {
  // clear and prepare output files
  for (auto pair = output_files.begin(); pair != output_files.end(); pair++) {
    if (final_only && pair->first != FinalStat)
      continue;
    string filename = get<0>(pair->second);
    ofstream& fout = get<1>(pair->second);
    fout.open(filename, ofstream::out | ofstream::trunc);
    assert(fout.good());
    fout.close();
    fout.open(filename, ofstream::app);
    assert(fout.good());
  }
}

void Stat::addSizeInfo(long raw, long aligned) {
  raw_bytes += raw;
  aligned_bytes += aligned;
}

void Stat::printSizeInfo() {
  iprintf("Amplification Raw: %lld Total: %lld Ratio: %f",
      raw_bytes, aligned_bytes, (double) raw_bytes / aligned_bytes);
}

bool Stat::outputFileExists() {
  // check if output files all exist
  for (auto pair = output_files.begin(); pair != output_files.end(); pair++) {
    if (!get<2>(pair->second)) continue;
    string filename = get<0>(pair->second);
    ifstream fin(filename);
    printf("%s: %s\n", filename.c_str(), fin.good() ? "good" : "bad");
    if (!fin.good())
      return false;
    fin.close();
  }
  return true;
}

void Stat::analyzeStat() {
  analyzeKernelStat();
  analyzePCIeStat();
  analyzeEvcStat();
}

void Stat::analyzeKernelStat() {
  string stat_file_in = get<0>(output_files[KernelStat]);
  ifstream fin(stat_file_in);
  ofstream& fout = get<1>(output_files[FinalStat]);
  int curit = 0, line_no = 0;
  string line;
  assert(fin.good());
  assert(fout.good());

  vector<string> stats;
  long curit_in_transfer = 0, curit_cpu_pf = 0, curit_ssd_pf = 0, curit_unalloc_pf = 0;
  long total_in_transfer = 0, total_cpu_pf = 0, total_ssd_pf = 0, total_unalloc_pf = 0;
  long curit_exe_time = 0, curit_ideal_exe_time = 0, curit_pf_exe_time = 0;
  long total_exe_time = 0, total_ideal_exe_time = 0, total_pf_exe_time = 0;
  std::ostringstream out_str;
  uint64_t last_e_time = 0;
  while (getline(fin, line)) {
    int num = getAllNumbersInLine(line, stats);
    line_no++;
    if (num != 1 && num != 9 && num != 12) {
      eprintf("Invalid line <%s> in stat file <%s:%d>, abort\n",
          line.c_str(), stat_file_in.c_str(), line_no);
      warn_corrupt_stat_file(stat_file_in);
      assert(false);
    }

    int iter = stod(stats[0]);
    if (iter != curit) {
      // conclude one iter
      double curit_sld = (double) curit_exe_time / curit_ideal_exe_time;
      double curit_spu = (double) curit_pf_exe_time / curit_exe_time;
      out_str.str("");
      out_str << "kernel_stat.iter" << curit << ".in_transfer = " << curit_in_transfer << "\n";
      out_str << "kernel_stat.iter" << curit << ".cpu_pf = " << curit_cpu_pf << "\n";
      out_str << "kernel_stat.iter" << curit << ".ssd_pf = " << curit_ssd_pf << "\n";
      out_str << "kernel_stat.iter" << curit << ".unalloc_pf = " << curit_unalloc_pf << "\n";
      out_str << "kernel_stat.iter" << curit << ".exe_time = " << curit_exe_time << "\n";
      out_str << "kernel_stat.iter" << curit << ".ideal_exe_time = " << curit_ideal_exe_time << "\n";
      out_str << "kernel_stat.iter" << curit << ".pf_exe_time = " << curit_pf_exe_time << "\n";
      out_str << "kernel_stat.iter" << curit << ".slowdown = " << curit_sld << "\n";
      out_str << "kernel_stat.iter" << curit << ".speedup = " << curit_spu << "\n";
      printf("%s", out_str.str().c_str());
      fout << out_str.str();
      total_in_transfer += curit_in_transfer; curit_in_transfer = 0;
      total_cpu_pf += curit_cpu_pf; curit_cpu_pf = 0;
      total_ssd_pf += curit_ssd_pf; curit_ssd_pf = 0;
      total_unalloc_pf += curit_unalloc_pf; curit_unalloc_pf = 0;
      total_exe_time += curit_exe_time; curit_exe_time = 0;
      total_ideal_exe_time += curit_ideal_exe_time; curit_ideal_exe_time = 0;
      total_pf_exe_time += curit_pf_exe_time; curit_pf_exe_time = 0;
      curit = iter;
    }
    if (iter == -1) {
      // conclude all analysis
      double total_sld = (double) total_exe_time / total_ideal_exe_time;
      double total_spu = (double) total_pf_exe_time / total_exe_time;
      out_str.str("");
      out_str << "kernel_stat.total.in_transfer = " << total_in_transfer << "\n";
      out_str << "kernel_stat.total.cpu_pf = " << total_cpu_pf << "\n";
      out_str << "kernel_stat.total.ssd_pf = " << total_ssd_pf << "\n";
      out_str << "kernel_stat.total.unalloc_pf = " << total_unalloc_pf << "\n";
      out_str << "kernel_stat.total.exe_time = " << total_exe_time << "\n";
      out_str << "kernel_stat.total.ideal_exe_time = " << total_ideal_exe_time << "\n";
      out_str << "kernel_stat.total.pf_exe_time = " << total_pf_exe_time << "\n";
      out_str << "kernel_stat.total.slowdown = " << total_sld << "\n";
      out_str << "kernel_stat.total.speedup = " << total_spu << "\n";
      printf("%s", out_str.str().c_str());
      fout << out_str.str();
      return;
    }
    // 0: iteration number        1: kernel id
    // 2: start time              3. end time
    // 4. ideal execution time    5. page-faulted execution time
    // 6. in transfer page number 7.CPU page fault number
    // 8. SSD page fault number   9. unalloc page fault number
    // [10]. CPU page used        [11]. GPU page used
    long s_time = stol(stats[2]);
    long e_time = stol(stats[3]);
    long exe_time = e_time - last_e_time;
    last_e_time = e_time;
    double ideal_sld = exe_time / stof(stats[4]);

    curit_in_transfer += stoi(stats[6]);
    curit_cpu_pf += stoi(stats[7]);
    curit_ssd_pf += stoi(stats[8]);
    curit_unalloc_pf += stoi(stats[9]);
    curit_exe_time += exe_time;
    curit_ideal_exe_time += stol(stats[4]);
    curit_pf_exe_time += stol(stats[5]);
  }
  if (line_no == 0)
    warn_corrupt_stat_file(stat_file_in);
}

void Stat::analyzePCIeStat() {
  string stat_file_in = get<0>(output_files[PCIeStat]);
  ifstream fin(stat_file_in);
  ofstream& fout = get<1>(output_files[FinalStat]);
  int curit = 0, line_no = 0;
  string line;
  assert(fin.good());
  assert(fout.good());

  vector<string> stats;
  long curit_alloc = 0, total_alloc = 0;
  long curit_incoming_pg = 0, curit_incoming_pg_ssd = 0, curit_incoming_pg_cpu = 0;
  long total_incoming_pg = 0, total_incoming_pg_ssd = 0, total_incoming_pg_cpu = 0;
  long curit_outgoing_pg = 0, curit_outgoing_pg_ssd = 0, curit_outgoing_pg_cpu = 0;
  long total_outgoing_pg = 0, total_outgoing_pg_ssd = 0, total_outgoing_pg_cpu = 0;
  std::ostringstream out_str;
  // TODO: fix kernel start time in the future
  uint64_t last_e_time = 0;
  while (getline(fin, line)) {
    int num = getAllNumbersInLine(line, stats);
    line_no++;
    if (num != 1 && num != 9) {
      eprintf("Invalid line <%s> in stat file <%s:%d>, abort\n",
          line.c_str(), stat_file_in.c_str(), line_no);
      warn_corrupt_stat_file(stat_file_in);
      assert(false);
    }

    int iter = stod(stats[0]);
    if (iter != curit) {
      // conclude one iter
      out_str.str("");
      out_str << "pcie_stat.iter" << curit << ".alloc = " << curit_alloc << "\n";
      out_str << "pcie_stat.iter" << curit << ".incoming_pg = " << curit_incoming_pg << "\n";
      out_str << "pcie_stat.iter" << curit << ".incoming_pg_cpu = " << curit_incoming_pg_cpu << "\n";
      out_str << "pcie_stat.iter" << curit << ".incoming_pg_ssd = " << curit_incoming_pg_ssd << "\n";
      out_str << "pcie_stat.iter" << curit << ".outgoing_pg = " << curit_outgoing_pg << "\n";
      out_str << "pcie_stat.iter" << curit << ".outgoing_pg_cpu = " << curit_outgoing_pg_cpu << "\n";
      out_str << "pcie_stat.iter" << curit << ".outgoing_pg_ssd = " << curit_outgoing_pg_ssd << "\n";
      printf("%s", out_str.str().c_str());
      fout << out_str.str();
      total_alloc += curit_alloc; curit_alloc = 0;
      total_incoming_pg += curit_incoming_pg; curit_incoming_pg = 0;
      total_incoming_pg_cpu += curit_incoming_pg_cpu; curit_incoming_pg_cpu = 0;
      total_incoming_pg_ssd += curit_incoming_pg_ssd; curit_incoming_pg_ssd = 0;
      total_outgoing_pg += curit_outgoing_pg; curit_outgoing_pg = 0;
      total_outgoing_pg_cpu += curit_outgoing_pg_cpu; curit_outgoing_pg_cpu = 0;
      total_outgoing_pg_ssd += curit_outgoing_pg_ssd; curit_outgoing_pg_ssd = 0;
      curit = iter;
    }
    if (iter == -1) {
      // conclude all analysis
      out_str.str("");
      out_str << "pcie_stat.total.alloc = " << total_alloc << "\n";
      out_str << "pcie_stat.total.incoming_pg = " << total_incoming_pg << "\n";
      out_str << "pcie_stat.total.incoming_pg_cpu = " << total_incoming_pg_cpu << "\n";
      out_str << "pcie_stat.total.incoming_pg_ssd = " << total_incoming_pg_ssd << "\n";
      out_str << "pcie_stat.total.outgoing_pg = " << total_outgoing_pg << "\n";
      out_str << "pcie_stat.total.outgoing_pg_cpu = " << total_outgoing_pg_cpu << "\n";
      out_str << "pcie_stat.total.outgoing_pg_ssd = " << total_outgoing_pg_ssd << "\n";
      printf("%s", out_str.str().c_str());
      fout << out_str.str();
      return;
    }
    // 0: iteration number        1: timestamp
    // 2: alloc page number
    // 3. incoming total page num 4. incoming SSD page num
    // 5. incoming CPU page num
    // 6. outgoing total page num 7. outgoing SSD page num
    // 8. outgoing CPU page num
    long time = stol(stats[1]);
    curit_alloc += stol(stats[2]);
    curit_incoming_pg += stol(stats[3]);
    curit_incoming_pg_ssd += stol(stats[4]);
    curit_incoming_pg_cpu += stol(stats[5]);
    curit_outgoing_pg += stol(stats[6]);
    curit_outgoing_pg_ssd += stol(stats[7]);
    curit_outgoing_pg_cpu += stol(stats[8]);
  }
  if (line_no == 0)
    warn_corrupt_stat_file(stat_file_in);
}

void Stat::analyzeEvcStat() {
  string stat_file_in = get<0>(output_files[EvcStat]);
  ifstream fin(stat_file_in);
  ofstream& fout = get<1>(output_files[FinalStat]);
  int curit = 0, line_no = 0;
  long timestamp, current_timestamp = -1;
  string line;
  assert(fin.good());
  assert(fout.good());

  vector<string> stats;
  map<Eviction_P, long> curit_hotness, total_hotness;
  std::ostringstream out_str;
  while (getline(fin, line)) {
    int num = getAllNumbersInLine(line, stats);
    line_no++;
    if (num != 1 && num != 7) {
      eprintf("Invalid line <%s> in stat file <%s:%d>, abort\n",
          line.c_str(), stat_file_in.c_str(), line_no);
      warn_corrupt_stat_file(stat_file_in);
      assert(false);
    }

    int iter = stoi(stats[0]);
    if (iter != curit) {
      // conclude one iter
      long total_pf = 0;
      out_str.str("");
      for (auto it = curit_hotness.begin(); it != curit_hotness.end(); ++it) {
        total_pf += it->second;
        total_hotness[it->first] += it->second;
      }
      out_str << "evc_stat.iter" << curit << ".total_evc = " << total_pf << "\n";
      for (auto it = curit_hotness.begin(); it != curit_hotness.end(); ++it) {
        out_str << "evc_stat.iter" << curit << "." << print_eviction_array[it->first].c_str() <<
            " = " << it->second << "\n";
        out_str << "evc_stat.iter" << curit << "." << print_eviction_array[it->first].c_str() <<
            "% = " << it->second * 100.0 / total_pf << "%\n";
      }
      printf("%s", out_str.str().c_str());
      fout << out_str.str();
      curit_hotness.clear();
      curit = iter;
    }
    if (iter == -1) {
      // conclude all analysis
      long total_pf = 0;
      for (auto it = total_hotness.begin(); it != total_hotness.end(); ++it) {
        total_pf += it->second;
      }
      out_str << "evc_stat.total.total_evc = " << total_pf << "\n";
      for (auto it = total_hotness.begin(); it != total_hotness.end(); ++it) {
        out_str << "evc_stat.total." << print_eviction_array[it->first].c_str() <<
            " = " << it->second << "\n";
        out_str << "evc_stat.total." << print_eviction_array[it->first].c_str() <<
            "% = " << it->second * 100.0 / total_pf << "%\n";
      }
      printf("%s", out_str.str().c_str());
      fout << out_str.str();
      return;
    }
    // 0: iteration number      1: kernel id
    // 2: timestamp             3. eviction destination
    // 4. vpn                   5. tensor id
    // 6. hotness
    Eviction_P hotness = static_cast<Eviction_P>(stoi(stats[6]));
    curit_hotness[hotness]++;
  }
  if (line_no == 0)
    warn_corrupt_stat_file(stat_file_in);
}

int Stat::getAllNumbersInLine(const string& input, vector<string>& output) const {
  const static auto is_numerical = [](char c) {
    return (c >= '0' && c <= '9') || c == '.' || c == '-';
  };
  output.clear();
  int idx = 0;
  const int size = input.size();
  while (idx < size) {
    if (is_numerical(input[idx])) {
      int start_idx = idx++;
      while (idx < size && is_numerical(input[idx])) idx++;
      output.push_back(input.substr(start_idx, idx - start_idx));
    } else {
      idx++;
    }
  }
  return output.size();
}

void Stat::warn_corrupt_stat_file(const string &file) const {
  wprintf("The content of stat file %s is corrupted. "
    "If simulation is not correctly finished, delete the simulation output folder and start a new simulation.\n",
    file.c_str());
}

// Stat END ========================

} // namespace Simulator
