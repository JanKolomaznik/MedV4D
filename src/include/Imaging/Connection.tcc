#ifndef _CONNECTION_H
#error File Connection.tcc cannot be included directly!
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
		ConnectProducerTyped( *port );
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::ConnectProducerTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::OutputImagePort &outputPort 
	)
{
	outputPort.PlugTyped( *this );
	_input = &outputPort;
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectIn()
{
	//TODO
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::ConnectConsumer( InputPort& inputPort )
{
	InputImagePort *port = dynamic_cast< InputImagePort * >( &inputPort );
	if( port ) {
		ConnectConsumerTyped( *port );
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::DisconnectOut( InputPort& inputPort )
{
	InputImagePort *port = dynamic_cast< InputImagePort * >( &inputPort );
	if( port ) {
		DisconnectOut( *port );
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::ConnectConsumerTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

	if( _consumers.find( inputPort.GetID() )== _consumers.end() ) {
		inputPort.PlugTyped( *this );
		_consumers[ inputPort.GetID() ] = &inputPort;
	} else {
		//TODO throw exception
	}

}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectOutTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

	//TODO
}


template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectAll()
{

	//TODO
}

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
		//TODO throw exception
	}
	//TODO
	CallImageFactoryRealloc( *_image, minimums, maximums, elementExtents );
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
	//TODO REWRITE !!!
	typename ConsumersMap::iterator it;
	for( it = _consumers.begin(); it != _consumers.end(); ++it ) {
		it->second->ReceiveMessage( msg, sendStyle, direction );
	}

}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_CONNECTION_H*/

