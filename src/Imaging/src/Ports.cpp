#include <algorithm>

#include "Thread.h"

#include "Functors.h"
#include "Imaging/Ports.h"

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
	//TODO handle special situations - messages for port, etc.
	if( _msgReceiver ) {
		_msgReceiver->ReceiveMessage( msg, sendStyle, direction );
	}
}

void
Port::UnPlug()
{
	_connection = NULL;
}
//******************************************************************************

bool 
Port::TryLockDataset()
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	} 

	return _connection->GetDataset().TryLockDataset();
}

void
Port::LockDataset()
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	} 

	_connection->GetDataset().LockDataset();
}

void 
Port::ReleaseDatasetLock()
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	} 

	_connection->GetDataset().UnlockDataset();
}

//******************************************************************************
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
//
//******************************************************************************

const AbstractImage &
InputPortAbstractImage
::GetAbstractImage()const
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	}
	return static_cast<ConnectionType*>( _connection )->GetAbstractImageReadOnly();
}


void
InputPortAbstractImage
::Plug( ConnectionInterface & connection )
{
	if( this->IsPlugged() ) {
		throw Port::EPortAlreadyConnected();
	}

	AbstractImageConnection *conn = 
		dynamic_cast< AbstractImageConnection * >( &connection );
	if( conn ) {
		this->_connection = conn;
	} else {
		throw Port::EConnectionTypeMismatch();
	}
}

//******************************************************************************

AbstractImage &
OutputPortAbstractImage
::GetAbstractImage()const
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	}
	return static_cast<ConnectionType*>( _connection )->GetAbstractImage();
}


void
OutputPortAbstractImage
::Plug( ConnectionInterface & connection )
{
	AbstractImageConnection *conn = 
		dynamic_cast< AbstractImageConnection * >( &connection );
	if( conn ) {
		this->_connection = conn;
	} else {
		throw Port::EConnectionTypeMismatch();
	}
}

//******************************************************************************

InputPortList::~InputPortList()
{
	std::for_each( 
		_ports.begin(), 
		_ports.end(), 
		Functors::Deletor< InputPort* >() 
		);
}

size_t
InputPortList::AddPort( InputPort* port )
{
	if( port == NULL ) {
		//TODO - throw exception
		return static_cast< size_t >( -1 );
	}
	_ports.push_back( port );
	port->SetReceiver( _msgReceiver );
	return _size++;
}

InputPort &
InputPortList::GetPort( size_t idx )const
{
	//TODO - check bounds - throw exception
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
		Functors::Deletor< OutputPort* >() 
		);
}

size_t
OutputPortList::AddPort( OutputPort* port )
{
	if( port == NULL ) {
		//TODO - throw exception
		return static_cast< size_t >( -1 );
	}
	_ports.push_back( port );
	port->SetReceiver( _msgReceiver );
	return _size++;
}

OutputPort &
OutputPortList::GetPort( size_t idx )const
{
	//TODO - check bounds - throw exception
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
