#ifndef _CONNECTION_H
#error File Connection.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

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

}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectOutTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::ConnectInTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::OutputImagePort &outputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectIn()
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectAll()
{

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
