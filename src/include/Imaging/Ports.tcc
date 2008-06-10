#ifndef _PORTS_H
#error File Ports.tcc cannot be included directly!
#else

#include "Imaging/Connection.h"

namespace M4D
{
namespace Imaging
{


template< typename ElementType, unsigned dimension >
const Image< ElementType, dimension >&
InputPortImageFilter< Image< ElementType, dimension > >
::GetImage()const
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	}
	return _imageConnection->GetImageReadOnly();
}


template< typename ElementType, unsigned dimension >
void
InputPortImageFilter< Image< ElementType, dimension > >
::Plug( ConnectionInterface & connection )
{
	ImageConnection< ImageType > *conn = 
		dynamic_cast< ImageConnection< ImageType > * >( &connection );
	if( conn ) {
		PlugTyped( *conn );
	} else {
		throw Port::EMismatchConnectionType();
	}
}

template< typename ElementType, unsigned dimension >
void
InputPortImageFilter< Image< ElementType, dimension > >
::PlugTyped( ImageConnection< Image< ElementType, dimension > > & connection )
{
	_imageConnection = &connection;
	//TODO
}

template< typename ElementType, unsigned dimension >
void
InputPortImageFilter< Image< ElementType, dimension > >
::UnPlug()
{
	//TODO
}

template< typename ElementType, unsigned dimension >
void
InputPortImageFilter< Image< ElementType, dimension > >
::SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		)
{
	if( this->IsPlugged() ) {
		msg->_senderID = this->GetID();
		_imageConnection->RouteMessage( msg, sendStyle, FD_AGAINST_FLOW );
		
	}
	//TODO
}

template< typename ElementType, unsigned dimension >
void
InputPortImageFilter< Image< ElementType, dimension > >
::ReceiveMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle, 
		FlowDirection				direction
		)
{
	//TODO
}

//******************************************************************************

template< typename ElementType, unsigned dimension >
Image< ElementType, dimension >&
OutputPortImageFilter< Image< ElementType, dimension > >
::GetImage()const
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	}

	return _imageConnection->GetImage();
}

template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::Plug( ConnectionInterface & connection )
{
	ImageConnection< ImageType > *conn = 
		dynamic_cast< ImageConnection< ImageType > * >( &connection );
	if( conn ) {
		PlugTyped( *conn );
	} else {
		throw Port::EMismatchConnectionType();
	}
}

template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::PlugTyped( ImageConnection< Image< ElementType, dimension > > & connection )
{
	_imageConnection = &connection;
	//TODO
}

template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::UnPlug()
{
	//TODO
}

template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		)
{
	if( this->IsPlugged() ) {
		msg->_senderID = this->GetID();
		_imageConnection->RouteMessage( msg, sendStyle, FD_IN_FLOW );
	}
	//TODO
}

template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::ReceiveMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle, 
		FlowDirection				direction
		)
{
	//TODO
}

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
