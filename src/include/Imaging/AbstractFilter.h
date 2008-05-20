#ifndef _ABSTRACT_FILTER_H
#define _ABSTRACT_FILTER_H

#include "Imaging/AbstractProcessingUnit.h"
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

	size_t
	GetInputPortCount()const;

	size_t
	GetOutputPortCount()const;

	virtual void
	SetInputConnection( ConnectionPtr conn, unsigned portIndex )=0;

	virtual void
	SetOutputConnection( ConnectionPtr conn, unsigned portIndex )=0;

	
	void
	Execute();

	void
	StopExecution();


	bool
	IsUpToDate();
protected:
	/**
	 * Method running in execution thread - this method wil be stopped, when
	 * StopExecution() is invoked.
	 * In inherited class reimplementation of this method is easy way to 
	 * implement new filter, and let all dirty work to ancestor class.
	 **/
	virtual void
	ExecutionThreadMethod()=0;

private:

};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_ABSTRACT_FILTER_H*/
