/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Pipeline.h 
 * @{ 
 **/

#ifndef _PIPELINE_H
#define _PIPELINE_H


#include "MedV4D/Imaging/APipeline.h"
#include "MedV4D/Imaging/ConnectionInterface.h"
#include <boost/shared_ptr.hpp>
#include <vector>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{


class Pipeline : public APipeline
{
public:
	
	Pipeline();

	~Pipeline();

	void
	AddFilter( APipeFilter *filter );

	void
	FillingFinished();

	/**
	 * Connect two compatible ports if possible.
	 * @param outPort Reference to output port of some filter.
	 * @param inPort Reference to input port of some filter.
	 **/
	ConnectionInterface &
	MakeConnection( M4D::Imaging::OutputPort& outPort, M4D::Imaging::InputPort& inPort );
protected:
	typedef std::vector< APipeFilter * > FilterVector;
	typedef std::vector< ConnectionInterface * > ConnectionVector;

	FilterVector		_filters;
	ConnectionVector	_connections;

private:

};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_PIPELINE_H*/


/** @} */

