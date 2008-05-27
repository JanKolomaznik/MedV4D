#include "Imaging/AbstractFilter.h"


namespace M4D
{
namespace Imaging
{

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
		new boost::thread( MainExecutionThread( this, ADAPTIVE_CALCULATION ) );
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
		new boost::thread( MainExecutionThread( this, RECALCULATION ) );
}

bool
AbstractFilter::StopExecution()
{
	//TODO
	
	return _workState.TryStop();
}

bool
AbstractFilter::CanContinue()
{
	//TODO
	return true;
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
