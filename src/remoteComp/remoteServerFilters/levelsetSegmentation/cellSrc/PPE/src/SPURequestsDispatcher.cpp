#include "common/Common.h"
#include "../../SPE/commonTypes.h"
#include "../SPURequestsDispatcher.h"

using namespace M4D::Cell;
using namespace M4D::Multithreading;

#define DEBUG_MAILBOX 12

///////////////////////////////////////////////////////////////////////////////

SPURequestsDispatcher::SPURequestsDispatcher(TWorkManager *wm, uint32 numSPE)
	: _workManager(wm), _numOfSPE(numSPE)
{
#ifdef FOR_CELL
	_SPE_data = new Tspu_pthread_data[_numOfSPE];
#else
	_progSims = new Tspu_prog_sim[_numOfSPE];
	
	for(uint32 i=0; i<_numOfSPE;i++)
	{
		_progSims[i]._applyUpdateCalc.m_layerGate._mailbox	= &_progSims[i]._mailbox;

		// setup apply update
		_progSims[i]._applyUpdateCalc.commonConf	= 
			&_workManager->GetConfSructs()[i].runConf;
		_progSims[i]._applyUpdateCalc.m_stepConfig = 
			&_workManager->GetConfSructs()[i].calcChngApplyUpdateConf;
		_progSims[i]._applyUpdateCalc.m_propLayerValuesConfig = 
			&_workManager->GetConfSructs()[i].propagateValsConf;

		// and update solver
		_progSims[i]._updateSolver.m_Conf = &_workManager->GetConfSructs()[i].runConf;
		_progSims[i]._updateSolver.m_stepConfig = 
			&_workManager->GetConfSructs()[i].calcChngApplyUpdateConf;
		//_progSims[i]._updateSolver.Init();
	}
#endif
	
	_results = new float32[_numOfSPE];
}

///////////////////////////////////////////////////////////////////////////////

SPURequestsDispatcher::~SPURequestsDispatcher()
{
#ifdef FOR_CELL
	delete [] _SPE_data;
#else
	delete [] _progSims;
#endif
	delete [] _results;
}

///////////////////////////////////////////////////////////////////////////////

float32 toFloat(uint32 val) { return *((float32 *) &val); }

void
SPURequestsDispatcher::DispatchMessage(uint32 i)
{
	uint32 dataRead = MyPopMessage(i);
		
	MessageID id = (MessageID) (dataRead & MessageID_MASK);
	switch (id)
	{
	case UNLINKED_NODES_PROCESS:
		DispatchUnlinkMessage(dataRead, i);
		break;
	case PUSHED_NODES_PROCESS:
		DispatchPushNodeMess(dataRead, i);
		break;
	case JOB_DONE:
		_results[i] = toFloat(MyPopMessage(i));
		_SPEYetRunning--;
		break;
	}
}
///////////////////////////////////////////////////////////////////////////////
void SPURequestsDispatcher::DispatchPushNodeMess(uint32 message, uint32 SPENum)
{
	uint8 lyerID = (message & MessageLyaerID_MASK) >> MessageLyaerID_SHIFT;
	uint32 param = (message & MessagePARAM_MASK) >> MessagePARAM_SHIFT;

	uint32 secondWord = MyPopMessage(SPENum);

	TIndex ind =
	{
	{ param, secondWord & 0xFFFF, (secondWord & (0xFFFF << 16)) >> 16 } };

	_workManager->PUSHNode(ind, lyerID);
	DL_PRINT(DEBUG_MAILBOX, "PUSH " << ind << " layr=" << (uint32)lyerID);
}

///////////////////////////////////////////////////////////////////////////////
void SPURequestsDispatcher::DispatchUnlinkMessage(uint32 message, uint32 SPENum)
{
	uint8 lyerID = (message & MessageLyaerID_MASK) >> MessageLyaerID_SHIFT;

	uint64 nodeAddress = MyPopMessage(SPENum);
	nodeAddress |= ((uint64) MyPopMessage(SPENum)) << 32;

	SparseFieldLevelSetNode *n = (SparseFieldLevelSetNode *) nodeAddress;

	_workManager->UNLINKNode(n, lyerID, SPENum);
	DL_PRINT(DEBUG_MAILBOX, "ULNK " << n << " layr=" << (uint32)lyerID);
}

///////////////////////////////////////////////////////////////////////////////

void SPURequestsDispatcher::MyPushMessage(uint32 message, uint32 SPENum)
{
#ifdef FOR_CELL
	if ((spe_in_mbox_write(_SPE_data[SPENum].spe_ctx, &message, 1,
			SPE_MBOX_ANY_NONBLOCKING)) == 0 )
		; //TODO except
#else
	_progSims[SPENum]._mailbox.PPEPush(message);
#endif
}

///////////////////////////////////////////////////////////////////////////////

uint32 SPURequestsDispatcher::MyPopMessage(uint32 SPENum)
{
#ifdef FOR_CELL
	uint32 dataRead;
	spe_out_mbox_read(_SPE_data[SPENum].spe_ctx, &dataRead, 1);
	D_PRINT("Read: " << dataRead);
	return dataRead;
#else
	return _progSims[SPENum]._mailbox.PPEPop();
#endif
}


///////////////////////////////////////////////////////////////////////////////

void SPURequestsDispatcher::SendCommand(ESPUCommands cmd)
{
	uint32 cmdData = (uint32) cmd;
	D_PRINT("Write command: " << cmd);
	for(uint32 i=0; i<_numOfSPE; i++)
	{
		MyPushMessage(cmdData, i);
		if(cmd == CALC_UPDATE)
		{
			// send dt param
			float32 dt = _workManager->_dt;
			MyPushMessage(*((uint32 *) &dt), i);
		}
	}
	
	WaitForMessages();
}

///////////////////////////////////////////////////////////////////////////////

void SPURequestsDispatcher::WaitForMessages()
{
	_SPEYetRunning = _numOfSPE;
	while(_SPEYetRunning)
	{
		for(uint32 i=0; i<_numOfSPE; i++)
		{
#ifdef FOR_CELL
			if(spe_out_mbox_status(_SPE_data[i].spe_ctx) < 1)
				DispatchMessage(i);
#else
			if(_progSims[i]._mailbox.spe_queue_status())
				DispatchMessage(i);
#endif
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_PC
void
Tspu_prog_sim::SimulateFunc()
{
	uint32 mailboxVal;
	float32 retval;
	do 
	{
		  mailboxVal = _mailbox.SPEPop();
		  switch( (ESPUCommands) mailboxVal)
		  {
		  case CALC_CHANGE:
			  //printf ("CALC_CHANGE received\n");
			  // calculate and return retval
			  retval = _updateSolver.CalculateChange();
			  {
			  			ScopedLock lock(_mailbox.fromSPEQMutex);
			  _mailbox.SPEPush((uint32) JOB_DONE);
			  _mailbox.SPEPush(*((uint32 *) &retval));
			  }
			  break;
		  case CALC_UPDATE:
			  //printf ("CALC_UPDATE received\n");
			  retval = _applyUpdateCalc.ApplyUpdate(toFloat(_mailbox.SPEPop()));
			  {
			  			ScopedLock lock(_mailbox.fromSPEQMutex);
			  _mailbox.SPEPush((uint32) JOB_DONE);
			  _mailbox.SPEPush(*((uint32 *) &retval));
			  }
			  break;
		  case CALC_PROPAG_VALS:
			  //printf ("CALC_UPDATE received\n");
			  _applyUpdateCalc.PropagateAllLayerValues();
			  {
			  			ScopedLock lock(_mailbox.fromSPEQMutex);
			  _mailbox.SPEPush((uint32) JOB_DONE);
			  _mailbox.SPEPush((uint32) retval);
			  }
			  break;
		  case QUIT:
			  printf ("QUIT received\n");
		  	  break;
		  }
	} while(mailboxVal != QUIT);
}
#endif
///////////////////////////////////////////////////////////////////////////////
