#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

/*
The simulator of G10 is part ly based on the UVMSmart and GPGPUSim simulators. (Including UVM migration latencies,
PCIe characteristics, Page replacement policies, far-fault managements, etc.)
https://github.com/DebashisGanguly/gpgpu-sim_UVMSmart
*/

#include <vector>
#include <deque>
#include <queue>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include "simulationComponents.h"
#include "simulationEvents.h"
#include "simulationUtils.h"
#include "analysis.h"
#include "ast.h"

using std::deque;
using std::string;
using std::vector;
using std::greater;
using std::unordered_map;
using std::unordered_set;
using std::priority_queue;

#define PAGE_SIZE (4096)


namespace Simulator {

/**
 * @brief Event simulator that operates on a message queue and process events on a time
 *        basis
 */
class EventSimulator {
  public:
    /**
     * @brief create a new event simulator
     * @param basename the output file that will be using to store the result
     * @note <stat_output_file>.kernel -- kernel related results
     *       <stat_output_file>.pcie   -- PCIe related results
     */
    EventSimulator(string basename);
    ~EventSimulator();

    /**
     * @brief step the simulation forward by parsing next event
     */
    void step();

    /**
     * @brief run one iteration (from the first kernel to last kernel) of simulation
     */
    void run();

    /**
     * @brief run several iterations of simulation
     * @param num_iter desired number of iteration to run
     */
    void run(unsigned num_iter);

  private:
    /**
     * @brief add an event to the event queue
     * @param event the event to be added
     */
    void schedule(Event *event);

    /**
     * @brief add a list of events to the event queue
     * @param events the list of events to be added
     */
    void schedule(vector<Event *> &events);

    // event queue
    priority_queue<EventPtr, vector<EventPtr>, EventPtrComp> event_queue;
    // simulation current time, updated before the actual events are run
    unsigned long current_time;
};

} // namespace Simulator

#endif
