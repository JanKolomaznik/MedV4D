#include "common/Common.h"
#include "../SPEManager.h"
#include <iostream>

#include <math.h> // sqrt function

using namespace M4D::Cell;
using namespace M4D::Multithreading;


///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
extern spe_program_handle_t SPEMain; // handle to SPE program

///////////////////////////////////////////////////////////////////////////////

void *ppu_pthread_function(void *arg)
{
	unsigned int entry = SPE_DEFAULT_ENTRY;

	Tppu_pthread_data *datap = (Tppu_pthread_data *)arg;

	std::cout << "Running SPE thread with param=" << datap->argp << std::endl;

	if (spe_context_run(datap->spe_ctx, &entry, 0, datap->argp, NULL, NULL) < 0)
	{
		perror("Failed running context");
		//exit (1);
	}
	pthread_exit(NULL);
}
#endif
///////////////////////////////////////////////////////////////////////////////

SPEManager::SPEManager()
{
	/* Determine the number of SPE threads to create.   */
	speCount = 4;//spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);

	{
		ScopedLock lock(SPURequestsDispatcher::mutexManagerTurn);
		DL_PRINT(DEBUG_SYNCHRO, "setting _managerTurn to true for lock the dispatchers");
		SPURequestsDispatcher::_managerTurn = true;
	}
	
	SPURequestsDispatcher::InitBarrier(speCount);

	m_requestDispatcher = new SPURequestsDispatcher[speCount];
	
	// run dispatchers
	for (uint32 i = 0; i< speCount; i++)
	{
		if (pthread_create(
			&m_requestDispatcher[i]._pthread, 
			NULL, 
			&ppu_pthread_function,
			&m_requestDispatcher[i])
			)
		{
			LOG("Failed creating thread");
		}
	}

#ifdef FOR_CELL
	data = new Tppu_pthread_data[speCount];
#endif
}

///////////////////////////////////////////////////////////////////////////////

void SPEManager::InitProgramProps(void)
{
	for (uint32 i = 0; i< speCount; i++)
	{
		m_requestDispatcher[i]._workManager = _workManager;

		m_requestDispatcher[i]._segmentID = i;
		m_requestDispatcher[i]._applyUpdateCalc.m_layerGate.dispatcher
				= &m_requestDispatcher[i];

		// setup apply update
		m_requestDispatcher[i]._applyUpdateCalc.commonConf
				= &_workManager->GetConfSructs()[i].runConf;
		m_requestDispatcher[i]._applyUpdateCalc.m_stepConfig
				= &_workManager->GetConfSructs()[i].calcChngApplyUpdateConf;
		m_requestDispatcher[i]._applyUpdateCalc.m_propLayerValuesConfig
				= &_workManager->GetConfSructs()[i].propagateValsConf;

		// and update solver
		m_requestDispatcher[i]._updateSolver.m_Conf
				= &_workManager->GetConfSructs()[i].runConf;
		m_requestDispatcher[i]._updateSolver.m_stepConfig
				= &_workManager->GetConfSructs()[i].calcChngApplyUpdateConf;
		m_requestDispatcher[i]._updateSolver.Init();
	}
}

///////////////////////////////////////////////////////////////////////////////

SPEManager::~SPEManager()
{
	//TODO stop the SPUs
#ifdef FOR_CELL
	ESPUCommands quitCommand = QUIT;
	SendCommand(quitCommand);

	// wait for thread termination
	for (uint32 i=0; i<speCount; i++)
	{
		if (pthread_join(data[i].pthread, NULL))
		{
			D_PRINT ("Failed joining thread");
		}
	}

	/* Destroy contexts */
	for (uint32 i=0; i<speCount; i++)
	{
		if (spe_context_destroy(data[i].spe_ctx) != 0)
		{
			D_PRINT("Failed destroying context");
			//exit (1);
		}
	}

	delete [] data;
	
#endif
	delete [] m_requestDispatcher;
}

///////////////////////////////////////////////////////////////////////////////

TimeStepType SPEManager::MergeTimesteps()
{
	// get minimum
	TimeStepType min = m_requestDispatcher[0]._result;
	for (uint32 i=1; i<speCount; i++)
	{
		if (m_requestDispatcher[i]._result < min)
			min = m_requestDispatcher[i]._result;
	}

	return min;
}

///////////////////////////////////////////////////////////////////////////////

TimeStepType SPEManager::MergeRMSs()
{
	TimeStepType accum = 0;

	// Determine the average change during this iteration.
	for (uint32 i=0; i<speCount; i++)
	{
		if (m_requestDispatcher[i]._result == 0)
			return 0;
		else
		{
			accum += m_requestDispatcher[i]._result;
		}
	}

	TimeStepType retval = sqrt(accum / _workManager->GetLayer0TotalSize() );

	return retval;
}

///////////////////////////////////////////////////////////////////////////////

void
SPEManager::RunDispatchers()
{
	{
		M4D::Multithreading::ScopedLock lock(SPURequestsDispatcher::doneCountMutex);
		SPURequestsDispatcher::_dipatchersYetWorking = speCount;
		DL_PRINT(DEBUG_SYNCHRO, "setting _dipatchersYetWorking to " << speCount);
	}
	
	//unblock the dispatchers
	{
		M4D::Multithreading::ScopedLock lock(SPURequestsDispatcher::mutexManagerTurn);
		DL_PRINT(DEBUG_SYNCHRO, "setting _managerTurn to false to unblock the dispatchers");
		SPURequestsDispatcher::_managerTurn = false;
	}
	// signal to dispatchers
	SPURequestsDispatcher::managerTurnValidCvar.notify_all();
	
	// wait until all dispatchers finish their command
	M4D::Multithreading::ScopedLock lock(SPURequestsDispatcher::doneCountMutex);
	DL_PRINT(DEBUG_SYNCHRO, "trying cond :" << 
			(SPURequestsDispatcher::_dipatchersYetWorking < speCount) );
    while(SPURequestsDispatcher::_dipatchersYetWorking > 0)
    {
    	SPURequestsDispatcher::doneCountCvar.wait(lock);
    	DL_PRINT(DEBUG_SYNCHRO, "trying cond :" << 
    					(SPURequestsDispatcher::_dipatchersYetWorking < speCount) );
    }
}

///////////////////////////////////////////////////////////////////////////////

TimeStepType
SPEManager::RunUpdateCalc()
{
	_workManager->AllocateUpdateBuffers();
	_workManager->InitCalculateChangeAndUpdActiveLayerConf();
	
	for (uint32 i = 0; i< speCount; i++)
				m_requestDispatcher[i]._command = CALC_CHANGE;

	RunDispatchers();

	return MergeTimesteps();
}

///////////////////////////////////////////////////////////////////////////////

double
SPEManager::ApplyUpdate(TimeStepType dt)
{
	_workManager->InitCalculateChangeAndUpdActiveLayerConf();
	_workManager->InitPropagateValuesConf();
	_workManager->_dt = dt;
	
	for (uint32 i = 0; i< speCount; i++)
		m_requestDispatcher[i]._command = CALC_UPDATE;

	RunDispatchers();
	
	return MergeRMSs();
}

///////////////////////////////////////////////////////////////////////////////

void
SPEManager::RunPropagateLayerVals()
{
	_workManager->InitPropagateValuesConf();
	
	for (uint32 i = 0; i< speCount; i++)
		m_requestDispatcher[i]._command = CALC_PROPAG_VALS;

	RunDispatchers();
}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
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

void SPEManager::RunSPEs(RunConfiguration *conf)
{
	for (uint32 i=0; i<speCount; i++)
	{
		/* Create SPE context */
		if ((data[i].spe_ctx = spe_context_create(0, NULL)) == NULL)
		{
			perror("Failed creating context");
			exit(1);
		}
		/* Load SPE program into the SPE context */
		if (spe_program_load(data[i].spe_ctx, &SPEMain))
		{
			perror("Failed loading program");
			exit(1);
		}
		/* Initialize context run data */
		data[i].argp = conf;
		/* Create pthread for each of the SPE conexts */
		if (pthread_create(&data[i].pthread, NULL, &ppu_pthread_function,
				&data[i]))
		{
			perror("Failed creating thread");
		}
	}
}
#endif
///////////////////////////////////////////////////////////////////////////////
