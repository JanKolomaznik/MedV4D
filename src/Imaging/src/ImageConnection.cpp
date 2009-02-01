/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageConnection.cpp 
 * @{ 
 **/

#include "Imaging/ImageConnection.h"

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

void
AbstractImageConnection
::ConnectConsumer( InputPort& inputPort )
{
	if( typeid(inputPort) == typeid( InputPortAbstractImage &) )
	{
		this->PushConsumer( inputPort );
	} else {
		_THROW_ ConnectionInterface::EMismatchPortType();
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
		_THROW_ ConnectionInterface::EMismatchPortType();
	}
}

void
AbstractImageConnection
::PutImage( M4D::Imaging::AbstractImage::Ptr image )
{
	if( !image ) {
		_THROW_ AbstractImageConnectionInterface::EInvalidImage();
	}
	this->_image = image;

	RouteMessage( 
			MsgDatasetPut::CreateMsg(), 
			PipelineMessage::MSS_NORMAL,
			FD_BOTH	
		);
}

const AbstractImage &
AbstractImageConnection
::GetAbstractImageReadOnly()const
{
	if( !_image ) {
		_THROW_ AbstractImageConnectionInterface::ENoImageAssociated();
	}
	
	return *_image;
}

AbstractImage &
AbstractImageConnection
::GetAbstractImage()const
{
	if( !_image ) {
		_THROW_ AbstractImageConnectionInterface::ENoImageAssociated();
	}
	
	return *_image;
}
	
}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

