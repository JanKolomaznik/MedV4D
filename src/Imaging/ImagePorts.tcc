/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImagePorts.tcc 
 * @{ 
 **/

#ifndef _IMAGE_PORTS_H
#error File ImagePorts.tcc cannot be included directly!
#else

#include "Imaging/ConnectionInterface.h"
#include "Imaging/ImageConnection.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

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
		_THROW_ EDisconnected( this->GetID() );
	}
	return static_cast<ConnectionType*>( _connection )->GetImageReadOnly();
}


template< typename ElementType, unsigned dimension >
void
InputPortImageFilter< Image< ElementType, dimension > >
::Plug( ConnectionInterface & connection )
{
	if( this->IsPlugged() ) {
		_THROW_ Port::EPortAlreadyConnected();
	}

	ImageConnection< ImageType > *conn = 
		dynamic_cast< ImageConnection< ImageType > * >( &connection );
	if( conn ) {
		this->_connection = conn;
		PortPluggedMsg();
	} else {
		_THROW_ Port::EConnectionTypeMismatch();
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
		_THROW_ EDisconnected( this->GetID() );
	}

	return static_cast<ConnectionType*>( _connection )->GetImage();
}

/*template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::SetImageSize( 
		uint32		dim,
		int32 		minimums[], 
		int32 		maximums[], 
		float32		elementExtents[]
	    )
{
	if( !this->IsPlugged() ) {
		_THROW_ EDisconnected( this->GetID() );
	}

	static_cast<ConnectionType*>( _connection )->SetImageSize( dim, minimums, maximums, elementExtents );	
}*/

template< typename ElementType, unsigned dimension >
void
OutputPortImageFilter< Image< ElementType, dimension > >
::Plug( ConnectionInterface & connection )
{
	if( this->IsPlugged() ) {
		_THROW_ Port::EPortAlreadyConnected();
	}

	ImageConnection< ImageType > *conn = 
		dynamic_cast< ImageConnection< ImageType > * >( &connection );
	if( conn ) {
		this->_connection = conn;
		PortPluggedMsg();
	} else {
		_THROW_ Port::EConnectionTypeMismatch();
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


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_IMAGE_PORTS_H*/

/** @} */

