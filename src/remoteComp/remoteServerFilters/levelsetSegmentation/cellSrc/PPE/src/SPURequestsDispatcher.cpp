#include "common/Common.h"
#include "../../SPE/commonTypes.h"
#include "../SPURequestsDispatcher.h"

using namespace M4D::Cell;
using namespace M4D::Multithreading;

Mutex SPURequestsDispatcher::mutexManagerTurn;
CondVar SPURequestsDispatcher::managerTurnValidCvar;
Mutex SPURequestsDispatcher::mutexDispatchersTurn;
CondVar SPURequestsDispatcher::doneCountCvar;
Mutex SPURequestsDispatcher::doneCountMutex;

Barrier *SPURequestsDispatcher::_barrier;

uint32 SPURequestsDispatcher::_dipatchersYetWorking;
bool SPURequestsDispatcher::_managerTurn;

#define DEBUG_MAILBOX 12

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void *ppu_pthread_function(void *arg)
{
	SPURequestsDispatcher *disp = (SPURequestsDispatcher*) arg;
	disp->DispatcherThreadFunc();
	
	pthread_exit(NULL);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

uint32
SPURequestsDispatcher::DispatcherThreadFunc()
{
	while( (WaitForCommand()) != QUIT)
	{		
		switch (_command)
		{
		case CALC_PROPAG_VALS:
			_applyUpdateCalc.PropagateAllLayerValues();
			break;
		case CALC_CHANGE:
			_updateSolver.UpdateFunctionProperties();
			_result = _updateSolver.CalculateChange();
			break;
		case CALC_UPDATE:
			_result = _applyUpdateCalc.ApplyUpdate(_workManager->_dt);
			break;
		default:
			ASSERT(false);
		}
		CommandDone();	// signal command is done
		_barrier->wait();	// wait for others
	}
	return 0;
}

///////////////////////////////////////////////////////////////////////////////

ESPUCommands
SPURequestsDispatcher::WaitForCommand()
{
	DL_PRINT(DEBUG_SYNCHRO, "waiting for command ...");
	
	ScopedLock lock(mutexManagerTurn);	// wait until mutex is unlocked by SPE manager
	while(_managerTurn)
		managerTurnValidCvar.wait(lock);
	
	return _command;
}

///////////////////////////////////////////////////////////////////////////////

void
SPURequestsDispatcher::CommandDone()
{
	{
		ScopedLock lock(doneCountMutex);
		_dipatchersYetWorking--;
		DL_PRINT(DEBUG_SYNCHRO, "decreasing _doneCount to " << _dipatchersYetWorking);
		if(_dipatchersYetWorking == 0)
		{
			// the last from crew give the turn to manager
			ScopedLock lock(mutexManagerTurn);
			_managerTurn = true;
			DL_PRINT(DEBUG_SYNCHRO, "setting _managerTurn=true");
		}
	}
	doneCountCvar.notify_all();
}

///////////////////////////////////////////////////////////////////////////////

void SPURequestsDispatcher::DispatchMessage(uint32 message)
{
	MessageID id = (MessageID) (message & MessageID_MASK);
	switch (id)
	{
	case UNLINKED_NODES_PROCESS:
		DispatchUnlinkMessage(message);
		break;
	case PUSHED_NODES_PROCESS:
		DispatchPushNodeMess(message);
		break;
	}
}
///////////////////////////////////////////////////////////////////////////////
void SPURequestsDispatcher::DispatchPushNodeMess(uint32 message)
{
	uint8 lyerID = (message & MessageLyaerID_MASK) >> MessageLyaerID_SHIFT;
	uint32 param = (message & MessagePARAM_MASK) >> MessagePARAM_SHIFT;

	uint32 secondWord = MyPopMessage();

	TIndex ind =
	{
	{ param, secondWord & 0xFFFF, (secondWord & (0xFFFF << 16)) >> 16 } };

	_workManager->PUSHNode(ind, lyerID);
	DL_PRINT(DEBUG_MAILBOX, "PUSH " << ind << " layr=" << (uint32)lyerID);
}

///////////////////////////////////////////////////////////////////////////////
void SPURequestsDispatcher::DispatchUnlinkMessage(uint32 message)
{
	uint8 lyerID = (message & MessageLyaerID_MASK) >> MessageLyaerID_SHIFT;

	uint64 nodeAddress = MyPopMessage();
	nodeAddress |= ((uint64) MyPopMessage()) << 32;

	LayerNodeType *n = (LayerNodeType *) nodeAddress;

	_workManager->UNLINKNode(n, lyerID, _segmentID);
	DL_PRINT(DEBUG_MAILBOX, "ULNK " << n << " layr=" << (uint32)lyerID);
}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_PC
void SPURequestsDispatcher::MyPushMessage(uint32 message)
{
	ScopedLock lock(mutex);
	messageQueue.push(message);
}

///////////////////////////////////////////////////////////////////////////////

uint32 SPURequestsDispatcher::MyPopMessage()
{
	ScopedLock lock(mutex);
	uint32 val = messageQueue.front();
	messageQueue.pop();
	return val;
}
#endif
///////////////////////////////////////////////////////////////////////////////
