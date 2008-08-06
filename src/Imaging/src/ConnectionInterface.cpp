#include "Imaging/ConnectionInterface.h"

namespace M4D
{
namespace Imaging
{

//*****************************************************************************

void
ConnectionInterface::DisconnectConsumer( InputPort& inputPort )
{
	ConsumersMap::iterator it = _consumers.find( inputPort.GetID() );
	if( it != _consumers.end() ) {
		inputPort.UnPlug();
		_consumers.erase( it );
	} else {
		//TODO throw exception
	}
}

void
ConnectionInterface::DisconnectProducer()
{
	if( _producer ) {
		_producer->UnPlug();
		_producer = NULL;
	} else {
		//TODO throw exception
	}
}

void
ConnectionInterface::DisconnectAll()
{
	//TODO
}

void
ConnectionInterface::PushConsumer( InputPort& consumer )
{
	if( _consumers.find( consumer.GetID() )== _consumers.end() ) {
		consumer.Plug( *this );
		_consumers[ consumer.GetID() ] = &consumer;
	} else {
		//TODO throw exception
	}
}


//*****************************************************************************

}/*namespace Imaging*/
}/*namespace M4D*/

