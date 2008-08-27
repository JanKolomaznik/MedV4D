#ifndef _ABSTRACT_PIPELINE_H
#define _ABSTRACT_PIPELINE_H

#include "Imaging/AbstractFilter.h"
#include "Imaging/ConnectionInterface.h"
#include "Imaging/Ports.h"
#include <boost/shared_ptr.hpp>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

class AbstractPipeline : public AbstractPipeFilter
{
public:
	struct Properties : public AbstractPipeFilter::Properties
	{

	};
	//TODO
	AbstractPipeline(): AbstractPipeFilter( new Properties ) {}

	virtual void
	AddFilter( AbstractPipeFilter *filter ) = 0;

	virtual void
	FillingFinished() = 0;

	/**
	 * Connect two compatible ports if possible.
	 * \param outPort Reference to output port of some filter.
	 * \param inPort Reference to input port of some filter.
	 **/
	virtual ConnectionInterface &
	MakeConnection( M4D::Imaging::OutputPort& outPort, M4D::Imaging::InputPort& inPort )=0;
protected:

private:

};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_PIPELINE_H*/
