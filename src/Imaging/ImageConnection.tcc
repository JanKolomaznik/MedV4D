#ifndef _IMAGE_CONNECTION_H
#error File ImageConnection.tcc cannot be included directly!
#else


namespace M4D
{
namespace Imaging
{

template< typename ElementType >
void 
CallImageFactoryRealloc( 
			Image< ElementType, 2 > &image,  
			size_t 		minimums[ 2 ], 
			size_t 		maximums[ 2 ], 
			float32		elementExtents[ 2 ]
	    )
{
	ImageFactory::ReallocateImage2DData< ElementType >( image, maximums[0]-minimums[0], maximums[1]-minimums[1] );
}

template< typename ElementType >
void 
CallImageFactoryRealloc( 
			Image< ElementType, 3 > &image,  
			size_t 		minimums[ 3 ], 
			size_t 		maximums[ 3 ], 
			float32		elementExtents[ 3 ]
	    )
{
	ImageFactory::ReallocateImage3DData< ElementType >( image, maximums[0]-minimums[0], maximums[1]-minimums[1], maximums[2]-minimums[2] );
}



//*****************************************************************************

template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::ConnectProducer( OutputPort& outputPort )
{
	OutputImagePort *port = 
		dynamic_cast< OutputImagePort * >( &outputPort );
	if( port ) {
			port->Plug( *this );
			_producer = port;
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::ConnectConsumer( InputPort& inputPort )
{
	//We allow only ports of exact type or generic port - nothing more!!!
	if( typeid(inputPort) == typeid( InputImagePort &) ||
		typeid(inputPort) == typeid( InputPortAbstractImage &)
	  )
	{
		this->PushConsumer( inputPort );
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

/*template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::DisconnectConsumer( InputPort& inputPort )
{
	InputImagePort *port = dynamic_cast< InputImagePort * >( &inputPort );
	if( port ) {
		DisconnectConsumer( *port );
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}*/

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::SetImageSize( 
		size_t 		minimums[ dimension ], 
		size_t 		maximums[ dimension ], 
		float32		elementExtents[ dimension ]
	    )
{
	if( !_image ) {
		throw ENoImageAssociated();
	}
	//TODO - check if locking should be done here
	_image->UpgradeToExclusiveLock();

	CallImageFactoryRealloc( *_image, minimums, maximums, elementExtents );

	_image->DowngradeFromExclusiveLock();
}




template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::RouteMessage( 
	PipelineMessage::Ptr 			msg, 
	PipelineMessage::MessageSendStyle 	sendStyle, 
	FlowDirection				direction
	)
{
	typename ConsumersMap::iterator it;
	for( it = _consumers.begin(); it != _consumers.end(); ++it ) {
		it->second->ReceiveMessage( msg, sendStyle, direction );
	}

}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE_CONNECTION_H*/

