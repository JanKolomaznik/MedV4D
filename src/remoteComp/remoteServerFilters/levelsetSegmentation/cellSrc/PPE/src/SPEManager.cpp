#include "common/Common.h"
#include "../SPEManager.h"
#include <iostream>

#include <math.h> // sqrt function
using namespace M4D::Cell;
using namespace M4D::Multithreading;

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

void SPEManager::StartSPEs()
{
	for (uint32 i=1; i<speCount; i++)
	{
		/* Create SPE context */
		_requestDispatcher._SPE_data[i].spe_ctx = spe_context_create(0, NULL);
		if (_requestDispatcher._SPE_data[i].spe_ctx == NULL)
		{
			perror("Failed creating context");
			exit(1);
		}
		/* Load SPE program into the SPE context */
		if (spe_program_load(_requestDispatcher._SPE_data[i].spe_ctx, &SPEMain))
		{
			perror("Failed loading program");
			exit(1);
		}
		/* Initialize context run data */
		_requestDispatcher._SPE_data[i].argp
				= (void *) &_workManager->GetConfSructs()[i];

		/* Create pthread for each of the SPE conexts */
		if (pthread_create( &_requestDispatcher._SPE_data[i].pthread, 
		NULL, &spu_pthread_function, &_requestDispatcher._SPE_data[i]))
		{
			perror("Failed creating thread");
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

void SPEManager::StopSPEs()
{
	_requestDispatcher.SendCommand(QUIT);

	// wait for thread termination
	for (uint32 i=1; i<speCount; i++)
	{
		if (pthread_join(_requestDispatcher._SPE_data[i].pthread, NULL))
		{
			D_PRINT ("Failed joining thread");
		}

		/* Destroy context */
		if (spe_context_destroy(_requestDispatcher._SPE_data[i].spe_ctx) != 0)
		{
			D_PRINT("Failed destroying context");
			//exit (1);
		}
	}
}
#endif
///////////////////////////////////////////////////////////////////////////////

/* Determine the number of SPE threads to create.   */
uint32 SPEManager::speCount = 1;//spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);

///////////////////////////////////////////////////////////////////////////////
uint32 SPEManager::GetSPECount()
{
	return speCount;
}

///////////////////////////////////////////////////////////////////////////////

SPEManager::SPEManager(SPURequestsDispatcher::TWorkManager *wm) :
	_workManager(wm), _requestDispatcher(wm, speCount)
{
	//	{
	//		ScopedLock lock(SPURequestsDispatcher::mutexManagerTurn);
	//		DL_PRINT(DEBUG_SYNCHRO, "setting _managerTurn to true for lock the dispatchers");
	//		SPURequestsDispatcher::_managerTurn = true;
	//	}
	//	
	//	SPURequestsDispatcher::InitBarrier(speCount);
	//
	//	m_requestDispatcher = new SPURequestsDispatcher[speCount];
	//	
	//	for (uint32 i = 0; i< speCount; i++)
	//		m_requestDispatcher[i].Init(_workManager, i);
	//	
	//	// run dispatchers
	//	for (uint32 i = 0; i< speCount; i++)
	//	{
	//		if (pthread_create(
	//			&m_requestDispatcher[i]._pthread, 
	//			NULL, 
	//			&ppu_pthread_function,
	//			&m_requestDispatcher[i])
	//			)
	//		{
	//			LOG("Failed creating thread");
	//		}
	//	}
#ifdef FOR_CELL
	StartSPEs();
#else
	
#endif
}

///////////////////////////////////////////////////////////////////////////////

SPEManager::~SPEManager()
{
	//	// stop dispatchers
	//	for (uint32 i = 0; i< speCount; i++)
	//			m_requestDispatcher[i]._command = QUIT;
	//
	//	UnblockDispatchers();
	//	
	//	for (uint32 i = 0; i< speCount; i++)
	//		pthread_join(m_requestDispatcher[i]._pthread, NULL);
	//		
	//	delete [] m_requestDispatcher;
#ifdef FOR_CELL
	StopSPEs();
#else
	
#endif
}

///////////////////////////////////////////////////////////////////////////////

TimeStepType SPEManager::MergeTimesteps()
{
	// get minimum
	TimeStepType min = _requestDispatcher._results[0];
	for (uint32 i=1; i<speCount; i++)
	{
		if (_requestDispatcher._results[i] < min)
			min = _requestDispatcher._results[i];
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
		if (_requestDispatcher._results[i] == 0)
			return 0;
		else
		{
			accum += _requestDispatcher._results[i];
		}
	}

	TimeStepType retval = sqrt(accum / _workManager->GetLayer0TotalSize() );

	return retval;
}

///////////////////////////////////////////////////////////////////////////////

//void
//SPEManager::UnblockDispatchers()
//{
//	//unblock the dispatchers
//	{
//		M4D::Multithreading::ScopedLock lock(SPURequestsDispatcher::mutexManagerTurn);
//		DL_PRINT(DEBUG_SYNCHRO, "setting _managerTurn to false to unblock the dispatchers");
//		SPURequestsDispatcher::_managerTurn = false;
//	}
//	// signal to dispatchers
//	SPURequestsDispatcher::managerTurnValidCvar.notify_all();
//}
//
/////////////////////////////////////////////////////////////////////////////////
//
//void
//SPEManager::RunDispatchers()
//{
//	{
//		M4D::Multithreading::ScopedLock lock(SPURequestsDispatcher::doneCountMutex);
//		SPURequestsDispatcher::_dipatchersYetWorking = speCount;
//		DL_PRINT(DEBUG_SYNCHRO, "setting _dipatchersYetWorking to " << speCount);
//	}
//	
//	UnblockDispatchers();
//	
//	// wait until all dispatchers finish their command
//	M4D::Multithreading::ScopedLock lock(SPURequestsDispatcher::doneCountMutex);
//	DL_PRINT(DEBUG_SYNCHRO, "trying cond :" << 
//			(SPURequestsDispatcher::_dipatchersYetWorking < speCount) );
//    while(SPURequestsDispatcher::_dipatchersYetWorking > 0)
//    {
//    	SPURequestsDispatcher::doneCountCvar.wait(lock);
//    	DL_PRINT(DEBUG_SYNCHRO, "trying cond :" << 
//    					(SPURequestsDispatcher::_dipatchersYetWorking < speCount) );
//    }
//}

///////////////////////////////////////////////////////////////////////////////

TimeStepType SPEManager::RunUpdateCalc()
{
	_workManager->AllocateUpdateBuffers();
	_workManager->InitCalculateChangeAndUpdActiveLayerConf();

	_requestDispatcher.SendCommand(CALC_CHANGE);
	//	for (uint32 i = 0; i< speCount; i++)
	//				m_requestDispatcher[i]._command = CALC_CHANGE;
	//
	//	RunDispatchers();

	return MergeTimesteps();
}

///////////////////////////////////////////////////////////////////////////////

double SPEManager::ApplyUpdate(TimeStepType dt)
{
	_workManager->InitCalculateChangeAndUpdActiveLayerConf();
	_workManager->InitPropagateValuesConf();
	_workManager->_dt = dt;

	//	for (uint32 i = 0; i< speCount; i++)
	//		m_requestDispatcher[i]._command = CALC_UPDATE;
	//
	//	RunDispatchers();
	_requestDispatcher.SendCommand(CALC_UPDATE);

	return MergeRMSs();
}

///////////////////////////////////////////////////////////////////////////////

void SPEManager::RunPropagateLayerVals()
{
	_workManager->InitPropagateValuesConf();

	//	for (uint32 i = 0; i< speCount; i++)
	//		m_requestDispatcher[i]._command = CALC_PROPAG_VALS;
	//
	//	RunDispatchers();
	_requestDispatcher.SendCommand(CALC_PROPAG_VALS);
}

///////////////////////////////////////////////////////////////////////////////

