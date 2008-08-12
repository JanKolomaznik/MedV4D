#include "Imaging/ImageConnection.h"

namespace M4D
{
namespace Imaging
{

//******************************************************************************

void
AbstractImageConnection
::ConnectConsumer( InputPort& inputPort )
{
	if( typeid(inputPort) == typeid( InputPortAbstractImage &) )
	{
		this->PushConsumer( inputPort );
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

void
AbstractImageConnection
::ConnectProducer( OutputPort& outputPort )
{
	OutputPortAbstractImage *port = 
		dynamic_cast< OutputPortAbstractImage * >( &outputPort );
	if( port ) {
			port->Plug( *this );
			_producer = port;
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

void
AbstractImageConnection
::PutImage( M4D::Imaging::AbstractImage::AImagePtr image )
{
	if( !image ) {
		throw AbstractImageConnectionInterface::EInvalidImage();
	}
	this->_image = image;

	RouteMessage( 
			MsgFilterUpdated::CreateMsg( true ), 
			PipelineMessage::MSS_NORMAL,
			FD_BOTH	
		);
}

const AbstractImage &
AbstractImageConnection
::GetAbstractImageReadOnly()const
{
	if( !_image ) {
		throw AbstractImageConnectionInterface::ENoImageAssociated();
	}
	
	return *_image;
}

AbstractImage &
AbstractImageConnection
::GetAbstractImage()const
{
	if( !_image ) {
		throw AbstractImageConnectionInterface::ENoImageAssociated();
	}
	
	return *_image;
}
	
}/*namespace Imaging*/
}/*namespace M4D*/

