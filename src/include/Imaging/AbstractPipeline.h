#ifndef _ABSTRACT_PIPELINE_H
#define _ABSTRACT_PIPELINE_H


#include "Imaging/AbstractFilter.h"
#include "Imaging/Connection.h"
#include <boost/shared_ptr.hpp>


namespace M4D
{
namespace Imaging
{

class AbstractPipeline : public AbstractPipeFilter
{
public:

	/**
	 * Connect two compatible ports if possible.
	 * @param outPort Reference to output port of some filter.
	 * @param inPort Reference to input port of some filter.
	 **/
	virtual Connection &
	MakeConnection( OutputPort& outPort, InputPort& inPort )=0;
protected:

private:

};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif _ABSTRACT_PIPELINE_H
