#ifndef _PORTS_H
#error File Ports.tcc cannot be included directly!
#else

#include "Imaging/Connection.h"

namespace M4D
{
namespace Imaging
{

//******************************************************************************

template< typename ElementType, unsigned dimension >
const Image< ElementType, dimension >&
InputPortImageFilter< Image< ElementType, dimension > >
::GetImage()const
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	}
	return static_cast<ConnectionType*>( _connection )->GetImageReadOnly();
}


template< typename ElementType, unsigned dimension >
void
InputPortImageFilter< Image< ElementType, dimension > >
::Plug( ConnectionInterface & connection )
{
	ImageConnection< ImageType > *conn = 
		dynamic_cast< ImageConnection< ImageType > * >( &connection );
	if( conn ) {
		this->_connection = conn;
	} else {
		throw Port::EConnectionTypeMismatch();
	}
}


/*
template< typename ElementType, unsigned dimension >
void
InputPortImageFilter< Image< ElementType, dimension > >
::SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		)
{
	if( this->IsPlugged() ) {
		DL_PRINT( 5, "Sending message " << msg->msgID );
		msg->senderID = this->GetID();
		static_cast<ConnectionType*>( _connection )->RouteMessage( msg, sendStyle, FD_AGAINST_FLOW );
		
	}
	//TODO
}
*/

//******************************************************************************

template< typename ElementType, unsigned dimension >
Image< ElementType, dimension >&
OutputPortImageFilter< Image< ElementType, dimension > >
::GetImage()const
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	}

	return static_cast<ConnectionType*>( _connection )->GetImage();
}

template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::SetImageSize( 
		size_t 		minimums[ dimension ], 
		size_t 		maximums[ dimension ], 
		float32		elementExtents[ dimension ]
	    )
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	}

	static_cast<ConnectionType*>( _connection )->SetImageSize( minimums, maximums, elementExtents );	
}

template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::Plug( ConnectionInterface & connection )
{
	ImageConnection< ImageType > *conn = 
		dynamic_cast< ImageConnection< ImageType > * >( &connection );
	if( conn ) {
		this->_connection = conn;
	} else {
		throw Port::EConnectionTypeMismatch();
	}
}


/*
template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		)
{
	if( this->IsPlugged() ) {
		DL_PRINT( 5, "Sending message " << msg->msgID );
		msg->senderID = this->GetID();
		static_cast<ConnectionType*>( _connection )->RouteMessage( msg, sendStyle, FD_IN_FLOW );
	}
	//TODO
}*/

//******************************************************************************

template< typename PortType >
PortType&
InputPortList::GetPortTyped( size_t idx )const
{
	return static_cast< PortType& >( GetPort( idx ) );
}

template< typename PortType >
PortType*
InputPortList::GetPortTypedSafe( size_t idx )const
{
	return dynamic_cast< PortType* >( &(GetPort( idx )) );
}

template< typename PortType >
PortType&
OutputPortList::GetPortTyped( size_t idx )const
{
	return static_cast< PortType& >( GetPort( idx ) );
}

template< typename PortType >
PortType*
OutputPortList::GetPortTypedSafe( size_t idx )const
{
	return dynamic_cast< PortType* >( &(GetPort( idx )) );
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_PORTS_H*/
