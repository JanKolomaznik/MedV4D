/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImagePorts.cpp 
 * @{ 
 **/

#include "Imaging/ImagePorts.h"

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

	AbstractImageConnectionInterface *conn = 
		dynamic_cast< AbstractImageConnectionInterface * >( &connection );
	if( conn ) {
		this->_connection = conn;
		PortPluggedMsg();
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
	AbstractImageConnectionInterface *conn = 
		dynamic_cast< AbstractImageConnectionInterface * >( &connection );
	if( conn ) {
		this->_connection = conn;
		PortPluggedMsg();
	} else {
		throw Port::EConnectionTypeMismatch();
	}
}

void
OutputPortAbstractImage
::SetImageSize( 
		uint32		dim,
		int32 		minimums[], 
		int32 		maximums[], 
		float32		elementExtents[]
	    )
{
	if( !this->IsPlugged() ) {
		throw EDisconnected( this->GetID() );
	}

	static_cast<AbstractImageConnectionInterface*>( _connection )->SetImageSize( dim, minimums, maximums, elementExtents );	
}
	
}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

