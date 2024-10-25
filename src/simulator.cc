#include <math.h>
#include <fstream>
#include <string>
#include <cstring>
#include <random>
#include <algorithm>
#include <math.h>
#include <climits>
#include <iostream>
#include <unistd.h>
#include "simulationComponents.h"
#include "simulationEvents.h"
#include "simulator.h"
#include "printUtils.h"

using std::pow;
using std::ceil;
using std::sort;
using std::pair;
using std::round;
using std::string;
using std::vector;
using std::ofstream;
using std::unordered_set;
using Simulator::TensorMovementHint;
using Simulator::TensorLocation;

extern vector<Tensor*> tensor_list;
extern vector<CUDAKernel> kernel_list;
// extern vector<Model_Layer*> forward_layers;
extern vector<TensorMovementHint> movement_hints;

extern long long memory_offset_intermediate;
extern long long memory_offset_weights;

bool is_ideal;

namespace Simulator {

extern System *sim_sys;
extern Stat *sim_stat;

EventSimulator::EventSimulator(string basename) {
  sim_sys = new Simulator::System();
  sim_stat = new Simulator::Stat(basename);
  sim_stat->prepareOutputFiles();

  // assert(memory_offset_intermediate < (long long) LONG_MAX);
  // assert(memory_offset_weights < (long long) LONG_MAX);
  // assert(memory_offset_intermediate + memory_offset_weights < (long long) LONG_MAX);

  // assert(isPageAligned(memory_offset_intermediate));
  // assert(isPageAligned(memory_offset_weights));

  long ideal_exe_time = 0;
  for (CUDAKernel kernel : kernel_list)
    ideal_exe_time += kernel.execution_cycles;
  printf("Ideal Execution Time: %ld cycles\n", ideal_exe_time);

  // Initialize CPU page table
  printf("Initializing CPU Page Table\n");
  for (Tensor *tensor : tensor_list) {
    Addr starting_addr = tensor->getGlobalOffset();
    long size_in_byte = (long) tensor->size_in_byte;
    long total_pages = ceil((double) size_in_byte / PAGE_SIZE);
    for (long page_num = 0; page_num < total_pages; page_num++) {
      Addr page_starting_addr = starting_addr + PAGE_SIZE * page_num;
      // create and get entry
      CPUPageTable::CPUPageTableEntry *entry =
          sim_sys->CPU_PT.createEntry(page_starting_addr);
      assert(entry);
      entry->ppn = 0;
      if (!is_ideal) {
        if (tensor->is_global_weight) {
          entry->location = IN_SSD; // UVM enabled, tensor started in SSD
        } else {
          entry->location = NOT_PRESENT; // UVM enabled, tensor started unallocated
        }
      } else {
        entry->location = IN_GPU; // ideal case, all tensor in GPU
      }
      entry->in_transfer = false;
    }
  }
  current_time = 0;

  // push initial events to event queue
  printf("Initial Events\n");
  vector<Event *> initial_events;
  initial_events.push_back(new KernelBeginEvent(current_time, sim_sys->getCurrentKernel()));
  initial_events.push_back(new BatcherEvent(current_time));
  schedule(initial_events);
  printf("Initial Events END #<%ld>\n", event_queue.size());
  printf("Simulation Start =============================================================\n");
}

EventSimulator::~EventSimulator() {
  sim_stat->printSizeInfo();
  delete sim_sys;
  delete sim_stat;
}

  // if (strncmp(event->name.c_str(), "Kernel", 6) == 0)
void EventSimulator::step() {
  vector<Event *> created_events;
  // get currently scheduled event
  Event *scheduled_event = event_queue.top().ptr;
  // advance time
  // if (current_time != scheduled_event->scheduled_time) {
  //   printf("Time advanced from <%15ld> to <%15ld> ===================\n",
  //       current_time, scheduled_event->scheduled_time);
  // }
  // assert(current_time <= scheduled_event->scheduled_time);
  current_time = scheduled_event->scheduled_time;
  // check if event is should be executed
  if (scheduled_event->shouldExecute()) {
    if (dynamic_cast<KernelBeginEvent *>(scheduled_event))
      printf("Executing: %s @ %ld [Duration:%ld] [ITER: %d] TotEvt:%ld\n",
          scheduled_event->name.c_str(), current_time,
          dynamic_cast<KernelBeginEvent *>(scheduled_event)->kernel->execution_cycles,
          sim_sys->getCurrentIteration(),
          event_queue.size());
    // execute current event
    scheduled_event->execute(created_events);
    // pop current event from event queue and create new events
    event_queue.pop();
    schedule(created_events);
    delete scheduled_event;
  }
}

void EventSimulator::run() {
  int this_iter = sim_sys->getCurrentIteration();
  iprintf("Simulation [ITER: %d] run starts @ %ld\n", this_iter, current_time);
  while (sim_sys->getCurrentIteration() == this_iter &&
         event_queue.size() > 0) {
    step();
  }
  iprintf("Simulation [ITER: %d] run ends @ %ld\n", this_iter, current_time);
}

void EventSimulator::run(unsigned num_iter) {
  printf("Simulation <%d> runs scheduled\n", num_iter);
  for (unsigned iter = 0; iter < num_iter; iter++)
    run();
}

void EventSimulator::schedule(Event *event) {
  if (event->name != "Exec")
  printf("  Scheduling: %s @ %ld\n",
      event->name.c_str(), event->scheduled_time);
  event_queue.emplace(EventPtr(event));
}

void EventSimulator::schedule(vector<Event *> &events) {
  for (Event *event : events)
    schedule(event);
}

} // namespace Simulator
