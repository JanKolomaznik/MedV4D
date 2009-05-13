#ifndef SPUREQUESTSDISPATCHER_H_
#define SPUREQUESTSDISPATCHER_H_



//to remove
#include "../supportClasses.h"

#include <pthread.h>


#include "workManager.h"

#define DEBUG_SYNCHRO 12

#ifdef FOR_CELL
#include <libspe2.h>
#else
#include "../SPE/updateCalculation/updateCalculatorSPE.h"
#include "../SPE/applyUpdateCalc/applyUpdateCalculator.h"
#include "mailboxSimulator.h"
#endif


//void *ppu_pthread_function(void *arg);

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
	UpdateCalculatorSPE _updateSolver;
	ApplyUpdateSPE _applyUpdateCalc;
	
	pthread_t pthread;
	MailboxSimulator _mailbox;
	
	void SimulateFunc(void);
};
#endif

class SPURequestsDispatcher
{
public:
	//typedef WorkManager<TIndex, float32> TWorkManager;
	
	
	SPURequestsDispatcher(WorkManager *wm, uint32 numSPE);
	~SPURequestsDispatcher();
	
	void DispatchMessage(uint32 SPENum);
	//	void SendResponse(uint32 res);

	void DispatchPushNodeMess(uint32 message, uint32 SPENum);
	void DispatchUnlinkMessage(uint32 message, uint32 SPENum);

	//typedef itk::Index<DIM> TIndex;

	
	//typedef itk::SparseFieldLevelSetNode<TIndex> LayerNodeType;
	
	//ESPUCommands WaitForCommand();
//	void CommandDone();
	
//	static M4D::Multithreading::Mutex mutexManagerTurn;
//	static M4D::Multithreading::CondVar managerTurnValidCvar;
//	static M4D::Multithreading::Mutex mutexDispatchersTurn;
//	static M4D::Multithreading::CondVar doneCountCvar;
//	static M4D::Multithreading::Mutex doneCountMutex;
//	static uint32 _dipatchersYetWorking;
//	static bool _managerTurn;
//	
//	static void InitBarrier(uint32 n) { _barrier = new M4D::Multithreading::Barrier(n); }
//	static M4D::Multithreading::Barrier *_barrier;

	WorkManager *_workManager;
	
//	uint32 DispatcherThreadFunc();
//	void Init(TWorkManager *wm, uint32 id);
	
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
