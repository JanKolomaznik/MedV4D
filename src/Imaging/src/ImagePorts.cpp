#include "Imaging/ImagePorts.h"

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
	} else {
		throw Port::EConnectionTypeMismatch();
	}
}

	
}/*namespace Imaging*/
}/*namespace M4D*/
