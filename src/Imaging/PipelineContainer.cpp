#include "Imaging/PipelineContainer.h"

#include <algorithm>

#include "Functors.h"

namespace M4D
{
namespace Imaging
{


PipelineContainer::PipelineContainer()
{

}

virtual
PipelineContainer::~PipelineContainer()
{
	stl::for_each( _connections.begin(), _connections.end(), Functors::Deletor< ConnectionInterface *>() );

	stl::for_each( _filters.begin(), _filters.end(), Functors::Deletor< AbstractPipeFilter *>() );
}

void
PipelineContainer::AddFilter( AbstractPipeFilter *filter )
{
	if( filter == NULL ) {
		throw ErrorHandling::ENULLPointer();
	}

	_filters.push_back( filter );
}

void
PipelineContainer::AddConnection( ConnectionInterface *connection )
{
	if( connection == NULL ) {
		throw ErrorHandling::ENULLPointer();
	}

	_connections.push_back( connection );
}

ConnectionInterface &
PipelineContainer::MakeConnection( M4D::Imaging::OutputPort& outPort, M4D::Imaging::InputPort& inPort )
{

}

ConnectionInterface &
PipelineContainer::MakeConnection( M4D::Imaging::AbstractPipeFilter& producer, unsigned producerPortNumber, 
		M4D::Imaging::AbstractPipeFilter& consumer, unsigned consumerPortNumber )
{

}

}/*namespace Imaging*/
}/*namespace M4D*/

