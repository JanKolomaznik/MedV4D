#include "Imaging/Pipeline.h"
#include <algorithm>
#include "Functors.h"

#include "Imaging/ImagePorts.h"
#include "Imaging/ImageConnection.h"

namespace M4D
{
namespace Imaging
{

ConnectionInterface *
CreateConnectionObjectFromPorts( OutputPort& outPort, InputPort& inPort )
{
	//TODO
	
	//checking if we have image ports
	if( dynamic_cast< OutputPortAbstractImage * > ( &outPort ) 
	  && dynamic_cast< InputPortAbstractImage * > ( &inPort ) ) {


	}

	return NULL;
}

Pipeline::Pipeline()
{
		//TODO

}

Pipeline::~Pipeline()
{
	std::for_each(
		_connections.begin(), 
		_connections.end(), 
		M4D::Functors::Deletor< ConnectionInterface* >() 
		);

	std::for_each(
		_filters.begin(), 
		_filters.end(), 
		M4D::Functors::Deletor< AbstractPipeFilter* >() 
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

ConnectionInterface &
Pipeline::MakeConnection( M4D::Imaging::OutputPort& outPort, M4D::Imaging::InputPort& inPort )
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
		//Newly created connection will be stored.
		_connections.push_back( connection );
	}

	connection->ConnectConsumer( inPort );
	connection->ConnectProducer( outPort );

	return *connection;
}


}/*namespace Imaging*/
}/*namespace M4D*/

