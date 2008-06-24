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
/**
 * Functor used in execution thread of filter - it's friend of AbstractPipeFilter.
 **/
struct MainExecutionThread
{
	/**
	 * @param filter Pointer to filter which executed thread with this 
	 * functor. MUST BE VALID!.
	 * @param updateType Which execution method of the filter will be invoked.
	 **/
	MainExecutionThread( 
		AbstractPipeFilter		*filter, 
		AbstractPipeFilter::UPDATE_TYPE	updateType 
		)
		: _filter( filter ), _updateType( updateType ) 
	{ /*empty*/ }

	/**
	 * Method executed by thread, which has copy of this object.
	 **/
	void
	operator()();
private:
	/**
	 * Filter which executed thread with this functor.
	 **/
	AbstractPipeFilter		*_filter;
	/**
	 * Type of execution method, which will be invoked.
	 **/
	AbstractPipeFilter::UPDATE_TYPE	_updateType;
};

void
MainExecutionThread::operator()()
{
	D_BLOCK_COMMENT( "++++ Entering MainExecutionThread()", "----- Leaving MainExecutionThread()" );

	//We want to do some steps before actual computing
	_filter->PreparationForComputing( _updateType );
	//TODO
	_filter->_outputPorts.SendMessage( MsgFilterStartModification::CreateMsg(), PipelineMessage::MSS_NORMAL );


	switch( _updateType ) {
	case AbstractPipeFilter::RECALCULATION:
		
		//Check how execution method finished its job.
		if( _filter->ExecutionOnWholeThreadMethod() ) {
			_filter->CleanAfterSuccessfulRun();
		} else {
			_filter->CleanAfterStoppedRun();
		}
		break;
	case AbstractPipeFilter::ADAPTIVE_CALCULATION:
		
		//Check how execution method finished its job.
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
AbstractPipeFilter::AbstractPipeFilter()
	: _inputPorts( this ), _outputPorts( this ), _invocationStyle( UIS_ON_DEMAND )
{

}

void
AbstractPipeFilter::Execute()
{
	//TODO
	/*TODO Call only when needed*/PrepareOutputDatasets();
	
	if( !_workState.TrySetRunning() ) {
		//TODO - handle
		return;
	}

	_executionThread =  
		new Multithreading::Thread( MainExecutionThread( this, ADAPTIVE_CALCULATION ) );
}

void
AbstractPipeFilter::ExecuteOnWhole()
{
	//TODO
	/*TODO Call only when needed*/PrepareOutputDatasets();

	if( !_workState.TrySetRunning() ) {
		//TODO - handle
		return;
	}
	_executionThread = 
		new Multithreading::Thread( MainExecutionThread( this, RECALCULATION ) );
}

bool
AbstractPipeFilter::StopExecution()
{
	//TODO
	
	return _workState.TrySetStopping();
}

bool
AbstractPipeFilter::CanContinue()
{
	//TODO
	return _workState.IsRunning();
}

void
AbstractPipeFilter::ReceiveMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle,
		FlowDirection				direction
		)
{
	//TODO
	/*switch ( sendStyle ) {
	}*/	
	switch( msg->msgID ) {
	case PMI_FILTER_UPDATED:
		InputDatasetUpdatedMsgHandler( static_cast< MsgFilterUpdated * >( msg.get() ) );
		break;
	case PMI_FILTER_START_MODIFICATION:
		InputDatasetStartModificationMsgHandler( static_cast< MsgFilterStartModification * >( msg.get() ) );
		break;
	default:
		//TODO	
		break;
	}
}

void
AbstractPipeFilter::InputDatasetUpdatedMsgHandler( MsgFilterUpdated *msg )
{
	//TODO - improve
	if( _invocationStyle == UIS_ON_UPDATE_FINISHED )
	{
		Execute();
	}
}

void
AbstractPipeFilter::InputDatasetStartModificationMsgHandler( MsgFilterStartModification *msg )
{
	//TODO - improve
	if( _invocationStyle == UIS_ON_CHANGE_BEGIN )
	{
		Execute();
	}
}

void
AbstractPipeFilter::CleanAfterSuccessfulRun()
{
	//TODO
	_outputPorts.SendMessage( MsgFilterUpdated::CreateMsg(), PipelineMessage::MSS_NORMAL );

	//We delete execution thread structure - thread will be detached.
	//Another execution thread can be created.
	delete _executionThread;
	//After this call no other changes to filter members are applied 
	//from this thread.
	_workState.SetUpToDate();
}

void
AbstractPipeFilter::CleanAfterStoppedRun()
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

