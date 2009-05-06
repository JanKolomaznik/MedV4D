#ifndef SPUREQUESTSDISPATCHER_H_
#define SPUREQUESTSDISPATCHER_H_

#include "common/Thread.h"

//to remove
#include "../supportClasses.h"

#include "itkIndex.h"

#include <pthread.h>
#include <queue>

#include "../SPE/updateCalculation/updateCalculatorSPE.h"
#include "../SPE/applyUpdateCalc/applyUpdateCalculator.h"

#include "workManager.h"

#define DEBUG_SYNCHRO 0

void *ppu_pthread_function(void *arg);

namespace M4D
{
namespace Cell
{

class SPURequestsDispatcher
{
public:
	void DispatchMessage(uint32 message);
	//	void SendResponse(uint32 res);

	void DispatchPushNodeMess(uint32 message);
	void DispatchUnlinkMessage(uint32 message);

	typedef itk::Index<DIM> TIndex;
	typedef std::queue<uint32> TMessageQueue;
	typedef WorkManager<TIndex, float32> TWorkManager;
	typedef itk::SparseFieldLevelSetNode<TIndex> LayerNodeType;
	
	ESPUCommands WaitForCommand();
	void CommandDone();

	//#define MAX_QUEUE_LEN 4
	TMessageQueue messageQueue;

	void MyPushMessage(uint32);
	uint32 MyPopMessage();

	M4D::Multithreading::Mutex mutex;
	
	static M4D::Multithreading::Mutex mutexManagerTurn;
	static M4D::Multithreading::CondVar managerTurnValidCvar;
	static M4D::Multithreading::Mutex mutexDispatchersTurn;
	static M4D::Multithreading::CondVar doneCountCvar;
	static M4D::Multithreading::Mutex doneCountMutex;
	static uint32 _dipatchersYetWorking;
	static bool _managerTurn;
	
	static void InitBarrier(uint32 n) { _barrier = new M4D::Multithreading::Barrier(n); }
	static M4D::Multithreading::Barrier *_barrier;

	TWorkManager *_workManager;
	
	uint32 DispatcherThreadFunc();
	
#ifdef FOR_CELL
	Tppu_pthread_data *data;
#else
	UpdateCalculatorSPE _updateSolver;
	ApplyUpdateSPE _applyUpdateCalc;
#endif

	TimeStepType _result;
	uint32 _segmentID;
	
	ESPUCommands _command;
	
	pthread_t _pthread;
};

}
}
#endif /*SPUREQUESTSDISPATCHER_H_*/
