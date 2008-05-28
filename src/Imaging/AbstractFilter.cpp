#include "Imaging/AbstractFilter.h"


namespace M4D
{
namespace Imaging
{

bool
FilterWorkingState::TrySetRunning()
{
	Multithreading::ScopedLock stateLock( _stateLock );
	switch( _state ) {
	case RUNNING:
	case STOPPING:
		return false;
	case UP_TO_DATE:
	case OUT_OF_DATE:
		_state = RUNNING;
		return true;
	default:
		//Shouldn't reach this.
		ASSERT( false );
	}
	return false;
}

bool
FilterWorkingState::TrySetStopping()
{
	Multithreading::ScopedLock stateLock( _stateLock );
	switch( _state ) {
	case RUNNING:
		_state = STOPPING;
	case STOPPING:
		return true;
	case UP_TO_DATE:
	case OUT_OF_DATE:
		return false;
	default:
		//Shouldn't reach this.
		ASSERT( false );
	}
	return false;
}

bool
FilterWorkingState::TrySetUpToDate()
{
	Multithreading::ScopedLock stateLock( _stateLock );
	switch( _state ) {
	case RUNNING:
	case STOPPING:
		return false;
	case UP_TO_DATE:
	case OUT_OF_DATE:
		_state = UP_TO_DATE;
		return true;
	default:
		//Shouldn't reach this.
		ASSERT( false );
	}
	return false;
}

bool
FilterWorkingState::TrySetOutOfDate()
{
	Multithreading::ScopedLock stateLock( _stateLock );
	switch( _state ) {
	case RUNNING:
	case STOPPING:
		return false;
	case UP_TO_DATE:
	case OUT_OF_DATE:
		_state = OUT_OF_DATE;
		return true;
	default:
		//Shouldn't reach this.
		ASSERT( false );
	}
	return false;
}



void
FilterWorkingState::SetRunning()
{
	Multithreading::ScopedLock stateLock( _stateLock );
	_state = RUNNING;
}

void
FilterWorkingState::SetStopping()
{
	Multithreading::ScopedLock stateLock( _stateLock );
	_state = STOPPING;
}

void
FilterWorkingState::SetUpToDate()
{
	Multithreading::ScopedLock stateLock( _stateLock );
	_state = UP_TO_DATE;
}

void
FilterWorkingState::SetOutOfDate()
{
	Multithreading::ScopedLock stateLock( _stateLock );
	_state = OUT_OF_DATE;
}

//******************************************************************************
struct MainExecutionThread
{
	MainExecutionThread( 
		AbstractFilter			*filter, 
		AbstractFilter::UPDATE_TYPE	updateType 
		)
		: _filter( filter ), _updateType( updateType ) 
	{ /*empty*/ }

	void
	operator()();
private:
	AbstractFilter			*_filter;
	AbstractFilter::UPDATE_TYPE	_updateType;
};

void
MainExecutionThread::operator()()
{
	switch( _updateType ) {
	case AbstractFilter::RECALCULATION:
		//Decide how execution method finished its job.
		if( _filter->ExecutionOnWholeThreadMethod() ) {
			_filter->CleanAfterSuccessfulRun();
		} else {
			_filter->CleanAfterStoppedRun();
		}
		break;
	case AbstractFilter::ADAPTIVE_CALCULATION:
		//Decide how execution method finished its job.
		if( _filter->ExecutionThreadMethod() ) {
			_filter->CleanAfterSuccessfulRun();
		} else {
			_filter->CleanAfterStoppedRun();
		}
		break;
	default:
		//Shouldn't reach this.
		ASSERT( false );
		break;
	}

}
//******************************************************************************
void
AbstractFilter::Execute()
{
	//TODO
	
	if( !_workState.TrySetRunning() ) {
		//TODO - handle
		return;
	}

	_executionThread =  
		new Multithreading::Thread( MainExecutionThread( this, ADAPTIVE_CALCULATION ) );
}

void
AbstractFilter::ExecuteOnWhole()
{
	//TODO

	if( !_workState.TrySetRunning() ) {
		//TODO - handle
		return;
	}
	_executionThread = 
		new Multithreading::Thread( MainExecutionThread( this, RECALCULATION ) );
}

bool
AbstractFilter::StopExecution()
{
	//TODO
	
	return _workState.TrySetStopping();
}

bool
AbstractFilter::CanContinue()
{
	//TODO
	return _workState.IsRunning();
}

void
AbstractFilter::CleanAfterSuccessfulRun()
{
	//We delete execution thread structure - thread will be detached.
	//Another execution thread can be created.
	delete _executionThread;
	//After this call no other changes to filter members are applied 
	//from this thread.
	_workState.SetUpToDate();
}

void
AbstractFilter::CleanAfterStoppedRun()
{
	//We delete execution thread structure - thread will be detached.
	//Another execution thread can be created.
	delete _executionThread;
	//After this call no other changes to filter members are applied 
	//from this thread.
	_workState.SetOutOfDate();
}

}/*namespace Imaging*/
}/*namespace M4D*/
