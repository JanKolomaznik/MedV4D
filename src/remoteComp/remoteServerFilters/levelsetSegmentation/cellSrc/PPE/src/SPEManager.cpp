#include "common/Common.h"
#include "../SPEManager.h"
#include <iostream>

#include <math.h> // sqrt function

using namespace M4D::Cell;
using namespace M4D::Multithreading;

/* Determine the number of SPE threads to create.   */
uint32 SPEManager::speCount = 1;//spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);

///////////////////////////////////////////////////////////////////////////////
uint32
SPEManager::GetSPECount()
{
	return speCount;
}

///////////////////////////////////////////////////////////////////////////////

SPEManager::SPEManager(SPURequestsDispatcher::TWorkManager *wm)
	: _workManager(wm)
{
	{
		ScopedLock lock(SPURequestsDispatcher::mutexManagerTurn);
		DL_PRINT(DEBUG_SYNCHRO, "setting _managerTurn to true for lock the dispatchers");
		SPURequestsDispatcher::_managerTurn = true;
	}
	
	SPURequestsDispatcher::InitBarrier(speCount);

	m_requestDispatcher = new SPURequestsDispatcher[speCount];
	
	for (uint32 i = 0; i< speCount; i++)
		m_requestDispatcher[i].Init(_workManager, i);
	
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
}

///////////////////////////////////////////////////////////////////////////////

SPEManager::~SPEManager()
{
	// stop dispatchers
	for (uint32 i = 0; i< speCount; i++)
			m_requestDispatcher[i]._command = QUIT;

	UnblockDispatchers();
	
	for (uint32 i = 0; i< speCount; i++)
		pthread_join(m_requestDispatcher[i]._pthread, NULL);
		
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
SPEManager::UnblockDispatchers()
{
	//unblock the dispatchers
	{
		M4D::Multithreading::ScopedLock lock(SPURequestsDispatcher::mutexManagerTurn);
		DL_PRINT(DEBUG_SYNCHRO, "setting _managerTurn to false to unblock the dispatchers");
		SPURequestsDispatcher::_managerTurn = false;
	}
	// signal to dispatchers
	SPURequestsDispatcher::managerTurnValidCvar.notify_all();
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
	
	UnblockDispatchers();
	
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

