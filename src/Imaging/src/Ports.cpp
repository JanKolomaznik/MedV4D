/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Ports.cpp 
 * @{ 
 **/

#include <algorithm>

#include "common/Thread.h"

#include "common/Functors.h"
#include "Imaging/Ports.h"
#include "Imaging/ConnectionInterface.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  @author Jan Kolomaznik
 */

namespace M4D
{
namespace Imaging
{

/* Port::PortID */ uint64
Port::GenerateUniqueID()
{
	static /* PortID */ uint64 lastID = 0;
	static Multithreading::Mutex genMutex;

	{	//We must synchronize to avoid multiple generation of one ID.
		Multithreading::ScopedLock lock( genMutex );
		return ++lastID;
	}
}
//******************************************************************************
	
void
Port::ReceiveMessage( 
	PipelineMessage::Ptr 			msg, 
	PipelineMessage::MessageSendStyle 	sendStyle, 
	FlowDirection				direction
	)
{
	ASSERT( _msgReceiver != NULL );
	//TODO handle special situations - messages for port, etc.
	if( _msgReceiver ) {
		
		DL_PRINT( 8, "PORT RECEIVING MESSAGE : Port=" << this << "; msgID=" << msg->msgID );
		
		_msgReceiver->ReceiveMessage( msg, sendStyle, direction );
	}
}

//******************************************************************************

bool 
Port::TryLockDataset()
{
	if( !this->IsPlugged() ) {
		_THROW_ EDisconnected( this->GetID() );
	} 

	return _connection->GetDataset().TryLockDataset();
}

void
Port::LockDataset()
{
	if( !this->IsPlugged() ) {
		_THROW_ EDisconnected( this->GetID() );
	} 

	_connection->GetDataset().LockDataset();
}

void 
Port::ReleaseDatasetLock()
{
	if( !this->IsPlugged() ) {
		_THROW_ EDisconnected( this->GetID() );
	} 

	_connection->GetDataset().UnlockDataset();
}

//******************************************************************************
void
InputPort::
UnPlug( bool onlyYourself )
{
	if( _connection == NULL ) {
		return;
	}

	//if should unplug only yourself, otherwise disconect connection.
	if( onlyYourself ) {
		_connection = NULL;
	} else {
		_connection->DisconnectConsumer( *this );
		_connection = NULL;
	}
}

void
InputPort
::SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		)
{
	if( this->IsPlugged() ) {
		msg->senderID = this->GetID();
		_connection->RouteMessage( msg, sendStyle, FD_AGAINST_FLOW );
	}
	//TODO
}

//******************************************************************************

void
OutputPort::
UnPlug( bool onlyYourself )
{
	if( _connection == NULL ) {
		return;
	}

	//if should unplug only yourself, otherwise disconect connection.
	if( onlyYourself ) {
		_connection = NULL;
	} else {
		_connection->DisconnectProducer();
		_connection = NULL;
	}
}

void
OutputPort::SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		)
{
	if( this->IsPlugged() ) {
		msg->senderID = this->GetID();
		_connection->RouteMessage( msg, sendStyle, FD_IN_FLOW );
	}
	//TODO
}

bool
InputPortTyped< AbstractDataSet >::IsConnectionCompatible( ConnectionInterface &conn )
{ 
	return dynamic_cast< IdealConnectionInterface * >( &conn ); 
}

bool
OutputPortTyped< AbstractDataSet >::IsConnectionCompatible( ConnectionInterface &conn )
{ 
	return dynamic_cast< IdealConnectionInterface * >( &conn ); 
}
//******************************************************************************
template< typename PortPointer >
struct PortFinisher
{
	void
	operator()( PortPointer port )
	{
		port->UnPlug();
		delete port;
	}
};
//******************************************************************************

InputPortList::~InputPortList()
{
	std::for_each( 
		_ports.begin(), 
		_ports.end(), 
		//Functors::Deletor< InputPort* >() 
		PortFinisher< InputPort* >()
		);
}

size_t
InputPortList::AppendPort( InputPort* port )
{
	if( port == NULL ) {
		//TODO - _THROW_ exception
		return static_cast< size_t >( -1 );
	}
	_ports.push_back( port );
	port->SetReceiver( _msgReceiver );
	return _size++;
}

InputPort &
InputPortList::GetPort( size_t idx )const
{
	if( _ports.size() <= idx ) {
		_THROW_ ErrorHandling::EBadIndex();
	}
	return *_ports[ idx ];
}

void
InputPortList::SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle
		)
{
	std::for_each( 
		_ports.begin(), 
		_ports.end(), 
		MessageSenderFunctor< InputPort* >( msg, sendStyle )
		);
}

OutputPortList::~OutputPortList()
{
	std::for_each( 
		_ports.begin(), 
		_ports.end(), 
		//Functors::Deletor< OutputPort* >() 
		PortFinisher< OutputPort* >()
		);
}

size_t
OutputPortList::AppendPort( OutputPort* port )
{
	if( port == NULL ) {
		//TODO - _THROW_ exception
		return static_cast< size_t >( -1 );
	}
	_ports.push_back( port );
	port->SetReceiver( _msgReceiver );
	return _size++;
}

OutputPort &
OutputPortList::GetPort( size_t idx )const
{
	if( _ports.size() <= idx ) {
		_THROW_ ErrorHandling::EBadIndex();
	}
	return *_ports[ idx ];
}

void
OutputPortList::SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle
		)
{
	std::for_each( 
		_ports.begin(), 
		_ports.end(), 
		MessageSenderFunctor< OutputPort* >( msg, sendStyle )
		);
}

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

