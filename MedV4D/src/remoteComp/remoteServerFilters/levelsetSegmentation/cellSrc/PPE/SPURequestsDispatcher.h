#ifndef SPUREQUESTSDISPATCHER_H_
#define SPUREQUESTSDISPATCHER_H_

//#include "../supportClasses.h"

#include <pthread.h>


#include "workManager.h"

#define DEBUG_SYNCHRO 12

#ifdef FOR_CELL
#include <libspe2.h>
#else
#include "../SPE/updateCalculation/updateCalculatorSPE.h"
#include "../SPE/applyUpdateCalc/applyUpdateCalculator.h"
#include "mailboxSimulator.h"
#include "../SPE/tools/sharedResources.h"
#endif

namespace M4D
{
namespace Cell
{

#ifdef FOR_CELL
struct Tspu_pthread_data
{
	spe_context_ptr_t spe_ctx;
	pthread_t pthread;
	void *argp;
};
#else
struct Tspu_prog_sim
{	
	pthread_t pthread;
	MailboxSimulator _mailbox;
	
	WorkManager *_wm;
	uint32 _speID;
	
	void SimulateFunc(void);
};
#endif

class SPURequestsDispatcher
{
public:
	
	
	SPURequestsDispatcher(WorkManager *wm, uint32 numSPE);
	~SPURequestsDispatcher();
	
	void DispatchMessage(uint32 SPENum);

	void DispatchPushNodeMess(uint32 message, uint32 SPENum);
	void DispatchUnlinkMessage(uint32 message, uint32 SPENum);

	WorkManager *_workManager;
	
	void MyPushMessage(uint32, uint32 SPENum);
	uint32 MyPopMessage(uint32 SPENum);
	
	float32 *_results;
	
	void SendCommand(ESPUCommands cmd);
	void WaitForMessages();
	
#ifdef FOR_CELL
	Tspu_pthread_data *_SPE_data;
#else
	Tspu_prog_sim *_progSims;
#endif
	uint32 _SPEYetRunning;
	uint32 _numOfSPE;
};

}
}
#endif /*SPUREQUESTSDISPATCHER_H_*/
