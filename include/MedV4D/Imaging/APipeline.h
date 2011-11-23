/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file APipeline.h 
 * @{ 
 **/

#ifndef _ABSTRACT_PIPELINE_H
#define _ABSTRACT_PIPELINE_H

#include "MedV4D/Imaging/AFilter.h"
#include "MedV4D/Imaging/ConnectionInterface.h"
#include "MedV4D/Imaging/Ports.h"
#include <boost/shared_ptr.hpp>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

class APipeline : public APipeFilter
{
public:
	struct Properties : public APipeFilter::Properties
	{

	};
	//TODO
	APipeline(): APipeFilter( new Properties ) {}

	virtual void
	AddFilter( APipeFilter *filter ) = 0;

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

/** @} */

