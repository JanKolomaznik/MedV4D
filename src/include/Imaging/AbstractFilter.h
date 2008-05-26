#ifndef _ABSTRACT_FILTER_H
#define _ABSTRACT_FILTER_H

#include "Imaging/AbstractProcessingUnit.h"
#include "Imaging/Ports.h"
#include "Imaging/Connection.h"
#include <boost/shared_ptr.hpp>


namespace M4D
{
namespace Imaging
{

class AbstractFilter : public AbstractProcessingUnit
{
public:
	typedef boost::shared_ptr< AbstractFilter > AbstractFilterPtr;

	virtual
	~AbstractFilter() {}

	const InputPortList &
	InputPort()const;

	const OutputPortList &
	OutputPort()const;
	
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
	 **/
	void
	StopExecution();

	bool
	IsUpToDate();
protected:
	/**
	 * Method running in execution thread - this method will be stopped, when
	 * StopExecution() is invoked.
	 * In inherited class reimplementation of this method is easy way to 
	 * implement new filter, and let all dirty work to ancestor class.
	 **/
	virtual void
	ExecutionThreadMethod()=0;


	InputPortList	_inputPorts;
	OutputPortList	_outputPorts;
private:

};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_ABSTRACT_FILTER_H*/
