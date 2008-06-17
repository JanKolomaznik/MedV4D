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
::ConnectIn( OutputPort& outputPort )
{
	OutputImagePort *port = 
		dynamic_cast< OutputImagePort * >( &outputPort );
	if( port ) {
		ConnectInTyped( *port );
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::ConnectInTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
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
::ConnectOut( InputPort& inputPort )
{
	InputImagePort *port = dynamic_cast< InputImagePort * >( &inputPort );
	if( port ) {
		ConnectOutTyped( *port );
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
::ConnectOutTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

	if( _outputs.find( inputPort.GetID() )== _outputs.end() ) {
		inputPort.PlugTyped( *this );
		_outputs[ inputPort.GetID() ] = &inputPort;
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
	CallImageFactoryRealloc( *_image, minimums, maximums );
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
	//TODO
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_CONNECTION_H*/

