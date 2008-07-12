#include "Imaging/Pipeline.h"
#include <algorithm>

namespace M4D
{
namespace Imaging
{

ConnectionInterface *CreateConnectionObjectFromPorts( outPort, inPort )
{
	//TODO
	return NULL;
}

Pipeline()
{
		//TODO

}

~Pipeline()
{
	std::for_each(
		_connections.begin(), 
		_connections.end(), 
		M4D::Functors::Deletor< Connection* > 
		);

	std::for_each(
		_filters.begin(), 
		_filters.end(), 
		M4D::Functors::Deletor< AbstractPipeFilter* > 
		);
}

void
Pipeline::AddFilter( AbstractPipeFilter *filter )
{
	if( filter == NULL ) {
		//TODO throw exception
	}
	_filters.push_back( filter );
}

void
Pipeline::FillingFinished()
{

		//TODO
}

Connection &
Pipeline::MakeConnection( OutputPort& outPort, InputPort& inPort )
{
	//if inPort occupied - error. Connection concept is designed only one to many.
	if( inPort.IsPlugged() ) {
		//TODO throw exception
	}

	ConnectionInterface *connection = NULL;
	//if outPort is connected, we use already created Conncetion, otherwise we 
	//have to create new one.
	if( outPort.IsPlugged() ) {
		//TODO -check
		connection = outPort.GetConnection();
	} else {

		//TODO
		connection = CreateConnectionObjectFromPorts( outPort, inPort );
	}

	connection->ConnectConsumer( inPort );
	connection->ConnectProducer( outPort );
}


}/*namespace Imaging*/
}/*namespace M4D*/

