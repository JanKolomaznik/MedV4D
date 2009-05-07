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
	
#ifdef FOR_CELL
	disp->StartSPE();
#endif
	
	disp->DispatcherThreadFunc();
	
#ifdef FOR_CELL
	disp->StopSPE();
#endif
	
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

void
SPURequestsDispatcher::Init(TWorkManager *wm, uint32 id)
{
	_workManager = wm;
	_segmentID = id;
	
	_applyUpdateCalc.m_layerGate.dispatcher	= this;

	// setup apply update
	_applyUpdateCalc.commonConf	= 
		&_workManager->GetConfSructs()[_segmentID].runConf;
	_applyUpdateCalc.m_stepConfig = 
		&_workManager->GetConfSructs()[_segmentID].calcChngApplyUpdateConf;
	_applyUpdateCalc.m_propLayerValuesConfig = 
		&_workManager->GetConfSructs()[_segmentID].propagateValsConf;

	// and update solver
	_updateSolver.m_Conf = &_workManager->GetConfSructs()[_segmentID].runConf;
	_updateSolver.m_stepConfig = 
		&_workManager->GetConfSructs()[_segmentID].calcChngApplyUpdateConf;
	_updateSolver.Init();
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

	SparseFieldLevelSetNode *n = (SparseFieldLevelSetNode *) nodeAddress;

	_workManager->UNLINKNode(n, lyerID, _segmentID);
	DL_PRINT(DEBUG_MAILBOX, "ULNK " << n << " layr=" << (uint32)lyerID);
}

///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// FOR_PC ///////////////////////////////////////
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
//////////////////////////////// FOR_CELL /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
extern spe_program_handle_t SPEMain; // handle to SPE program

///////////////////////////////////////////////////////////////////////////////

void *spu_pthread_function(void *arg)
{
	unsigned int entry = SPE_DEFAULT_ENTRY;

	Tspu_pthread_data *datap = (Tspu_pthread_data *)arg;

	std::cout << "Running SPE thread with param=" << datap->argp << std::endl;

	if (spe_context_run(datap->spe_ctx, &entry, 0, datap->argp, NULL, NULL) < 0)
	{
		perror("Failed running context");
		//exit (1);
	}
	pthread_exit(NULL);
}

///////////////////////////////////////////////////////////////////////////////

void
SPURequestsDispatcher::StopSPE()
{
	//TODO stop the SPUs

	ESPUCommands quitCommand = QUIT;
	SendCommand(quitCommand);

	// wait for thread termination
	if (pthread_join(_SPE_data.pthread, NULL))
	{
		D_PRINT ("Failed joining thread");
	}

	/* Destroy context */
	if (spe_context_destroy(_SPE_data.spe_ctx) != 0)
	{
		D_PRINT("Failed destroying context");
		//exit (1);
	}
	

}
///////////////////////////////////////////////////////////////////////////////

void SPEManager::SendCommand(enum ESPUCommands &cmd)
{
	uint32 result;
	for (uint32 i=0; i<speCount; i++)
	{
		D_PRINT("Write to SPE no: " << i << "'s mailbox, data=" << cmd);
		result = spe_in_mbox_write(data[i].spe_ctx, (uint32*) &cmd, 1,
				SPE_MBOX_ANY_NONBLOCKING);
		if (result == (uint32) -1)
			; //TODO except
	}
}

///////////////////////////////////////////////////////////////////////////////

void SPEManager::WaitForCommanResult()
{
	uint32 dataRead;
	for (uint32 i=0; i<speCount; i++)
	{
		D_PRINT("Read mailbox of " << i << "SPU, waiting...");
		while (spe_out_mbox_status(data[i].spe_ctx) < 1)
			;
		spe_out_mbox_read(data[i].spe_ctx, &dataRead, 1);
		D_PRINT("Read: " << dataRead);
	}
}

///////////////////////////////////////////////////////////////////////////////

void
SPEManager::StartSPE()
{
	/* Create SPE context */
	if ((_SPE_data.spe_ctx = spe_context_create(0, NULL)) == NULL)
	{
		perror("Failed creating context");
		exit(1);
	}
	/* Load SPE program into the SPE context */
	if (spe_program_load(_SPE_data.spe_ctx, &SPEMain))
	{
		perror("Failed loading program");
		exit(1);
	}
	/* Initialize context run data */
	_SPE_data.argp = _workManager->GetConfSructs()[_segmentID];
	/* Create pthread for each of the SPE conexts */
	if(pthread_create(
			&_SPE_data.pthread, NULL, &spu_pthread_function,	&_SPE_data))
	{
		perror("Failed creating thread");
	}
}
#endif
///////////////////////////////////////////////////////////////////////////////
