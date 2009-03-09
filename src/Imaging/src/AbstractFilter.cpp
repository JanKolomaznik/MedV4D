/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractFilter.cpp 
 * @{ 
 **/

#include "Imaging/AbstractFilter.h"
#include <ctime>

#ifdef _MSC_VER 
	#pragma warning (disable : 4355)
#endif

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  @author Jan Kolomaznik
 */

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
	D_PRINT( "++++ Filter " << _filter->GetName() << " = " << _filter ); 

	//TODO - handle exceptions

	//Set to default - we decide in BeforeComputation() method 
	//whether call PrepareOutputDatasets() method.
	_filter->_callPrepareOutputDatasets = false;

	try {
		//We check properties before we use them.
		D_PRINT( "++++++ " << _filter->GetName() << " - CheckProperties()" );
		_filter->_properties->CheckProperties();

		//We want to do some steps before actual computing
		D_PRINT( "++++++ " << _filter->GetName() << " - BeforeComputation()" );
		_filter->BeforeComputation( _updateType );
		
		//We decide whether resize output datasets
		if( _filter->_callPrepareOutputDatasets ) {
			D_PRINT( "++++++ " << _filter->GetName() << " - PrepareOutputDatasets()" );
			_filter->PrepareOutputDatasets();
			_filter->_callPrepareOutputDatasets = false;
		}
	} 
	catch( ErrorHandling::ExceptionBase &e ) {
		D_PRINT( "------ " << _filter->GetName() << " - EXCEPTION OCCURED : " << e );
		_filter->_outputPorts.SendMessage( 
			MsgFilterExecutionCanceled::CreateMsg(), PipelineMessage::MSS_NORMAL );
		_filter->CleanAfterStoppedRun();
		return;
	}
	catch( ... ) {
		D_PRINT( "------ " << _filter->GetName() << " - UNKNOWN EXCEPTION OCCURED : " );
		_filter->_outputPorts.SendMessage( 
			MsgFilterExecutionCanceled::CreateMsg(), PipelineMessage::MSS_NORMAL );
		_filter->CleanAfterStoppedRun();
		return;
	}


	//Mark changed parts of output
	D_PRINT( "++++++ " << _filter->GetName() << " - MarkChanges()" );
	_filter->MarkChanges( _updateType );

	_filter->_outputPorts.SendMessage( 
			MsgFilterStartModification::CreateMsg( _updateType == AbstractPipeFilter::RECALCULATION ), 
			PipelineMessage::MSS_NORMAL 
			);

	D_PRINT( "++++++ " << _filter->GetName() << " - ExecutionThreadMethod()" );
	clock_t time = clock();
	bool result = _filter->ExecutionThreadMethod( _updateType );
	time = clock() - time;
	LOG( _filter->GetName() << " was running for : " << (((float32)time)/CLOCKS_PER_SEC) << " seconds." );

	if( result ) {
		//Send message about finished job	
		_filter->_outputPorts.SendMessage( 
				MsgFilterUpdated::CreateMsg( _updateType == AbstractPipeFilter::RECALCULATION ), 
				PipelineMessage::MSS_NORMAL 
				);

		D_PRINT( "++++++ " << _filter->GetName() << " - AfterComputation( true )" );
		_filter->AfterComputation( true );
		_filter->CleanAfterSuccessfulRun();
	} else {
		//Send message about canceled job	
		_filter->_outputPorts.SendMessage( 
				MsgFilterExecutionCanceled::CreateMsg(), 
				PipelineMessage::MSS_NORMAL 
				);
		
		D_PRINT( "++++++ " << _filter->GetName() << " - AfterComputation( false )" );
		_filter->AfterComputation( false );

		_filter->CleanAfterStoppedRun();
	}

}
//******************************************************************************
AbstractPipeFilter::AbstractPipeFilter( AbstractPipeFilter::Properties *prop )
	: PredecessorType( prop ), _inputPorts( this ), _outputPorts( this ), 
	_invocationStyle( UIS_ON_DEMAND ), _propertiesTimestamp( Common::DefaultTimeStamp )
{

}

void
AbstractPipeFilter::Execute()
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
AbstractPipeFilter::ExecuteOnWhole()
{
	//TODO

	if( !_workState.TrySetRunning() ) {
		//TODO - handle
		return;
	}
	_executionThread = 
		new Multithreading::Thread( MainExecutionThread( this, RECALCULATION ) );
}

void
AbstractPipeFilter::ReleaseInputDataSet( uint32 idx )const
{
	_inputPorts[ idx ].ReleaseDatasetLock();
}

void
AbstractPipeFilter::ReleaseOutputDataSet( uint32 idx )const
{
	_outputPorts[ idx ].ReleaseDatasetLock();
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
	case PMI_FILTER_CANCELED:
		InputDatasetComputationCanceledMsgHandler( static_cast< MsgFilterExecutionCanceled * >( msg.get() ) );
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
		if( msg->IsUpdatedWhole() ) {
			ExecuteOnWhole();
		} else {
			Execute();
		}
	}
}

void
AbstractPipeFilter::InputDatasetComputationCanceledMsgHandler( MsgFilterExecutionCanceled *msg )
{
	//TODO
	StopExecution();
}

void
AbstractPipeFilter::InputDatasetStartModificationMsgHandler( MsgFilterStartModification *msg )
{
	//TODO - improve
	if( _invocationStyle == UIS_ON_CHANGE_BEGIN )
	{
		if( msg->IsUpdatedWhole() ) {
			ExecuteOnWhole();
		} else {
			Execute();
		}
	}
}

 void
AbstractPipeFilter::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{

}

void
AbstractPipeFilter::CleanAfterSuccessfulRun()
{

	//We delete execution thread structure - thread will be detached.
	//Another execution thread can be created.

	//TODO - check if right
	delete _executionThread;
	_executionThread = NULL;

	//After this call no other changes to filter members are applied 
	//from this thread.
	_workState.SetUpToDate();
}

void
AbstractPipeFilter::CleanAfterStoppedRun()
{
	//We delete execution thread structure - thread will be detached.
	//Another execution thread can be created.
	
	//TODO - check if right
	delete _executionThread;
	_executionThread = NULL;

	//After this call no other changes to filter members are applied 
	//from this thread.
	_workState.SetOutOfDate();
}

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

