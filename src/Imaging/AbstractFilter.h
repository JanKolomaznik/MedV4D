#ifndef _ABSTRACT_FILTER_H
#define _ABSTRACT_FILTER_H

#include "Common.h"
#include "Imaging/AbstractProcessingUnit.h"
#include "Imaging/Ports.h"
#include "Imaging/ConnectionInterface.h"
#include <boost/shared_ptr.hpp>
#include "Thread.h"
#include "Imaging/PipelineMessages.h"

#include "Imaging/filterProperties.h"

#include <iostream>
namespace M4D
{
namespace Imaging
{
 
#define GET_PROPERTIES_DEFINITION_MACRO \
	Properties & GetProperties(){ return *(static_cast<Properties*>( this->_properties ) ); }

#define GET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME ) \
	TYPE Get##NAME ()const{ return (static_cast<Properties*>( this->_properties ) )->PROPERTY_NAME ; }

#define SET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME ) \
	void Set##NAME ( TYPE value ){ (static_cast<Properties*>( this->_properties ) )->PROPERTY_NAME = value; }

#define GET_SET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME ) \
	GET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME ) \
	SET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME ) 
/**
 * Structure synchronizing access to filter state informations.
 **/
struct FilterWorkingState
{
public:
	/**
	 * Construct state object in UP_TO_DATE state.
	 **/
	FilterWorkingState():_state( UP_TO_DATE ) {}

	/**
	 * Try set state to RUNNING - only possible when in state 
	 * UP_TO_DATE or OUT_OF_DATE.
	 * @return Whether set was successful.
	 **/
	bool
	TrySetRunning();

	/**
	 * Try set state to STOPPING - only possible when in state 
	 * RUNNING or already STOPPING.
	 * @return Whether set was successful.
	 **/
	bool
	TrySetStopping();

	/**
	 * Try set state to UP_TO_DATE - only possible when in state 
	 * OUT_OF_DATE or already UP_TO_DATE.
	 * @return Whether set was successful.
	 **/
	bool
	TrySetUpToDate();
	
	/**
	 * Try set state to OUT_OF_DATE - only possible when in state 
	 * UP_TO_DATE or already OUT_OF_DATE.
	 * @return Whether set was successful.
	 **/
	bool
	TrySetOutOfDate();

	/**
	 * Change state unconditionaly.
	 **/
	void
	SetRunning();

	/**
	 * Change state unconditionaly.
	 **/
	void
	SetStopping();

	/**
	 * Change state unconditionaly.
	 **/
	void
	SetUpToDate();

	/**
	 * Change state unconditionaly.
	 **/
	void
	SetOutOfDate();

	/**
	 * @return True if is in state RUNNING.
	 **/
	bool
	IsRunning()const
		{ return _state == RUNNING; }
private:
	/**
	 * Enumeration of all possible states.
	 **/
	enum FILTER_STATE{ RUNNING, STOPPING, UP_TO_DATE, OUT_OF_DATE };

	/**
	 * Actual state.
	 **/
	FILTER_STATE		_state;

	/**
	 * Mutex for locking when changing state.
	 **/
	Multithreading::Mutex	_stateLock;
};

/**
 * Ancestor of all filters with basic execution logic.
 **/
class AbstractFilter 
  : public AbstractProcessingUnit
{
public:
	struct Properties
	{
		virtual 
		~Properties(){}
		virtual void
		CheckProperties() {}
	};
  
	/**
	 * Smart pointer to filter with this interface.
	 **/
	typedef boost::shared_ptr< AbstractFilter > AbstractFilterPtr;

	/**
	 * Destructor - virtual - can be polymorphically destroyed.
	 **/
	~AbstractFilter() { delete _properties; }
	
	/**
	 * Start computing only on modified data.
	 * Asynchronous method.
	 **/
	virtual void
	Execute() = 0;

	/**
	 * Start computing from scratch - recalculate output 
	 * using all input data, even when no change was applied.
	 * Asynchronous method.
	 **/
	virtual void
	ExecuteOnWhole() = 0;

	/**
	 * Stop execution of filter as soon as possible.
	 * Asynchronous method.
	 * \return True if stopping call was successful - it means that 
	 * filter will stop its execution. Otherwise some problem occured - 
	 * filter couldn't be stopped.
	 **/
	virtual bool
	StopExecution() = 0;

protected:

  	AbstractFilter( Properties * prop ): _properties( prop ) {}

	/**
	*  Filter's settings. Used to sending to server.
	*  This is pointer to base abstract settings class.
	*  !!! Each new filter derived from this class
	*  should declare new settings type derived from 
	*  FilterSettingTemplate class (filterProperties.h) 
	*  with template param of type FilterID (FilterIDEnums.h).
	*  This new enum item should be also added to enum with a new
	*  data set class !!!
	*/
	Properties *_properties;

	
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractFilter );
};

/**
 * Ancestor of all filters with basic execution logic.
 **/
class AbstractPipeFilter : public AbstractFilter, public MessageReceiverInterface
{
public:
	typedef AbstractFilter	PredecessorType;

	struct Properties: public PredecessorType::Properties
	{

	};

	enum UpdateInvocationStyle {
		UIS_ON_DEMAND,
		UIS_ON_CHANGE_BEGIN,
		UIS_ON_UPDATE_FINISHED		
	};

	/**
	 * Types of possible execution methods - used by execution thread functor.
	 **/
	enum UPDATE_TYPE{ RECALCULATION, ADAPTIVE_CALCULATION };
	/**
	 * Smart pointer to filter with this interface.
	 **/
	typedef boost::shared_ptr< AbstractPipeFilter > AbstractPipeFilterPtr;


	/**
	 * Destructor - virtual - can be polymorphically destroyed.
	 **/
	~AbstractPipeFilter() {}

	/**
	 * @return Returns list of all available input ports.
	 **/
	const InputPortList &
	InputPort()const
		{ return _inputPorts; }

	/**
	 * @return Returns list of all available output ports.
	 **/
	const OutputPortList &
	OutputPort()const
		{ return _outputPorts; }
	
	/**
	 * Start computing only on modified data.
	 * Asynchronous method.
	 **/
	void
	Execute();

	/**
	 * Start computing from scretch - recalculate output 
	 * using all input data, even when no change was applied.
	 * Asynchronous method.
	 **/
	virtual void
	ExecuteOnWhole();

	/**
	 * Stop execution of filter as soon as possible.
	 * Asynchronous method.
	 * \return True if stopping call was successful - it means that 
	 * filter will stop its execution. Otherwise some problem occured - 
	 * filter couldn't be stopped.
	 **/
	bool
	StopExecution();

	/**
	 * @return True if all computations are done or don't have input data.
	 **/
	bool
	IsUpToDate();

	unsigned
	SetParallelization( unsigned threadCount );

	unsigned
	GetParallelization()const;

	/**
	 * Sets ivocation style of filter - when it will be executed.
	 * \param style Style to be set.
	 **/
	void
	SetUpdateInvocationStyle( UpdateInvocationStyle style )
		{ _invocationStyle = style; }


	/**
	 * Method for receiving messages - called by sender.
	 * \param msg Smart pointer to message object - we don't have 
	 * to worry about deallocation.
	 * \param sendStyle How treat incoming message.
	 **/
	void
	ReceiveMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle,
		FlowDirection				direction
		);
	
protected:
	friend struct MainExecutionThread;


	AbstractPipeFilter( Properties *prop );
	
	/**
	 * Method running in execution thread - this method will be 
	 * stopped, when StopExecution() is invoked.
	 * In inherited class reimplementation of this method is easy way to 
	 * implement new filter, and let all dirty work to ancestor class.
	 * \return True if execution wasn't stopped, false otherwise.
	 **/
	virtual bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )=0;

	/**
	 * Method used for checking whether execution can continue.
	 * Execution threads should call this method often to ensure that 
	 * StopExecution() method wasn't called.
	 * \return True if computing can continue, false otherwise.
	 **/
	bool
	CanContinue();

	/**
	 * Method used for clean after successful run of execution thread.
	 * It will first detach thread and than set _workState to UP_TO_DATE.
	 * Because execution thread is detached, new execution thread can be 
	 * started. The old one will only finish - after this call it doesn't
	 * change internal state of filter.
	 **/
	void
	CleanAfterSuccessfulRun();

	/**
	 * Method used for clean after stopped run of execution thread.
	 * It will first detach thread and than set _workState to OUT_OF_DATE.
	 * Because execution thread is detached, new execution thread can be 
	 * started. The old one will only finish - after this call it doesn't
	 * change internal state of filter.
	 **/
	void
	CleanAfterStoppedRun();

	/**
	 * Method which will prepare datasets connected by output ports.
	 * Set their extents, etc.
	 **/
	virtual void
	PrepareOutputDatasets() { /*empty*/ };

	/**
	 * Method called in execution methods before actual computation.
	 * When overriding in successors predecessor implementation must be called first.
	 * \param utype Input/output parameter choosing desired update method. If 
	 * desired update method can't be used - right type is put as output value.
	 **/
	virtual void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	/**
	 * Method called in execution methods after computation.
	 * When overriding in successors, predecessor implementation must be called as last.
	 * \param successful Information, whether computation proceeded without problems.
	 **/
	virtual void
	AfterComputation( bool successful ){ /*empty*/ };

	void
	InputDatasetUpdatedMsgHandler( MsgFilterUpdated *msg );

	void
	InputDatasetStartModificationMsgHandler( MsgFilterStartModification *msg );

	void	
	InputDatasetComputationCanceledMsgHandler( MsgFilterExecutionCanceled *msg );

	/**
	 * Container for input ports - polymorphic interfaces.
	 **/
	InputPortList		_inputPorts;

	/**
	 * Container for output ports - polymorphic interfaces.
	 **/
	OutputPortList		_outputPorts;

	/**
	 * Main filter thread - computation is executed in this thread, 
	 * or its child threads.
	 **/
	Multithreading::Thread	*_executionThread;

	/**
	 * Internal state of filter.
	 **/
	FilterWorkingState	_workState;

	/**
	 * Invocation style of filter - see comments for possible values.
	 **/
	UpdateInvocationStyle 	_invocationStyle;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractPipeFilter );

};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_ABSTRACT_FILTER_H*/
