#include "Imaging/ConnectionInterface.h"

#include "Imaging/Ports.h"

namespace M4D
{
namespace Imaging
{

//*****************************************************************************

void
ConnectionInterface::RouteMessage( 
	PipelineMessage::Ptr 			msg, 
	PipelineMessage::MessageSendStyle 	sendStyle, 
	FlowDirection				direction
	)
{
	if( _messageHook ) {
		_messageHook->ReceiveMessage( msg, sendStyle, direction );
	}

	if( direction == FD_IN_FLOW ) {
		ConsumersMap::iterator it;
		for( it = _consumers.begin(); it != _consumers.end(); ++it ) {
			it->second->ReceiveMessage( msg, sendStyle, direction );
		}
	} else {
		if( _producer ) {
			_producer->ReceiveMessage( msg, sendStyle, direction );
		}

	}

}

void
ConnectionInterface::DisconnectConsumer( InputPort& inputPort )
{
	ConsumersMap::iterator it = _consumers.find( inputPort.GetID() );
	if( it != _consumers.end() ) {
		inputPort.UnPlug( true );
		_consumers.erase( it );
	} else {
		//TODO throw exception
	}
}

void
ConnectionInterface::DisconnectProducer()
{
	if( _producer ) {
		_producer->UnPlug( true );
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

