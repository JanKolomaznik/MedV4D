#ifndef _PIPELINE_CONTAINER_H
#define _PIPELINE_CONTAINER_H


#include "Imaging/AbstractFilter.h"
#include "Imaging/Connection.h"
#include <boost/shared_ptr.hpp>
#include <vector>

namespace M4D
{
namespace Imaging
{

class PipelineContainer
{
public:
	
	PipelineContainer();

	virtual
	~PipelineContainer();

	void
	AddFilter( AbstractPipeFilter *filter );

	void
	AddConnection( ConnectionInterface *connection );

	/**
	 * Connect two compatible ports if possible.
	 * @param outPort Reference to output port of some filter.
	 * @param inPort Reference to input port of some filter.
	 **/
	ConnectionInterface &
	MakeConnection( M4D::Imaging::OutputPort& outPort, M4D::Imaging::InputPort& inPort );

	ConnectionInterface &
	MakeConnection( M4D::Imaging::AbstractPipeFilter& producer, unsigned producerPortNumber, 
			M4D::Imaging::AbstractPipeFilter& consumer, unsigned consumerPortNumber );
protected:
	typedef std::vector< AbstractPipeFilter * > FilterVector;
	typedef std::vector< ConnectionInterface * > ConnectionVector;

	FilterVector		_filters;
	ConnectionVector	_connections;

private:

};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_PIPELINE_H*/

