#ifndef _ABSTRACT_FILTER_H
#define _ABSTRACT_FILTER_H

#include "Common.h"
#include "Imaging/AbstractProcessingUnit.h"
#include "Imaging/Ports.h"
#include "Imaging/Connection.h"
#include <boost/shared_ptr.hpp>
#include "Thread.h"


namespace M4D
{
namespace Imaging
{

/**
 * Structure synchronizing access to filter state informations.
 **/
struct FilterWorkingState
{
public:
	bool
	TrySetRunning();
	bool
	TrySetStopping();
	bool
	TrySetUpToDate();
	bool
	TrySetOutOfDate();

	bool
	TryStop();

	void
	SetRunning();
	void
	SetStopping();
	void
	SetUpToDate();
	void
	SetOutOfDate();

	bool
	IsRunning();
private:
	enum FILTER_STATE{ RUNNING, STOPPING, UP_TO_DATE, OUT_OF_DATE };

	FILTER_STATE	_state;
	boost::mutex	_stateLock;
};

/**
 * Ancestor of all filters with basic execution logic.
 **/
class AbstractFilter : public AbstractProcessingUnit
{
public:
	enum UPDATE_TYPE{ RECALCULATION, ADAPTIVE_CALCULATION };
	/**
	 * Smart pointer to filter with this interface.
	 **/
	typedef boost::shared_ptr< AbstractFilter > AbstractFilterPtr;

	virtual
	~AbstractFilter() {}

	/**
	 * \return Returns list of all available input ports.
	 **/
	const InputPortList &
	InputPort()const
		{ return _inputPorts; }

	/**
	 * \return Returns list of all available output ports.
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
	void
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

	bool
	IsUpToDate();
protected:
	friend struct MainExecutionThread;
	/**
	 * Method running in execution thread - this method or 
	 * ExecutionOnWholeThreadMethod() will be stopped, when
	 * StopExecution() is invoked.
	 * In inherited class reimplementation of this method is easy way to 
	 * implement new filter, and let all dirty work to ancestor class.
	 * \return True if execution wasn't stopped, false otherwise.
	 **/
	virtual bool
	ExecutionThreadMethod()=0;

	/**
	 * Method running in execution thread if ExecuteOnWhole() was called - 
	 * this method or ExecutionThreadMethod() will be stopped, when 
	 * StopExecution() is invoked.
	 * In inherited class reimplementation of this method is easy way to 
	 * implement new filter, and let all dirty work to ancestor class.
	 * \return True if execution wasn't stopped, false otherwise.
	 **/
	virtual bool
	ExecutionOnWholeThreadMethod()=0;

	/**
	 * Method used for checking wheather execution can continue.
	 * Execution threads should call this method often to ensure that 
	 * StopExecution() method wasn't called.
	 * \return True if computing can continue, false otherwise.
	 **/
	bool
	CanContinue();

	void
	CleanAfterSuccessfulRun();

	void
	CleanAfterStoppedRun();
	
	InputPortList		_inputPorts;
	OutputPortList		_outputPorts;
	boost::thread		*_executionThread;
	FilterWorkingState	_workState;
private:
	//Not implemented
	AbstractFilter( const AbstractFilter& );
	AbstractFilter&
	operator=( const AbstractFilter& );

};


}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_ABSTRACT_FILTER_H*/
