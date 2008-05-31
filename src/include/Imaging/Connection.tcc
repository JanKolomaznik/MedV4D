#ifndef _CONNECTION_H
#error File Connection.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename ElementType, unsigned dimension >
void
InImageConnection< Image< ElementType, dimension > >
::ConnectIn( OutputPort& outputPort )
{
	OutputImagePort *port = 
		dynamic_cast< OutputImagePort * >( &outputPort );
	if( port ) {
		ConnectIn( *port );
	} else {
		//TODO - throw exception
	}
}

template< typename ElementType, unsigned dimension >
void
OutImageConnection< Image< ElementType, dimension > >
::ConnectOut( InputPort& inputPort )
{
	InputImagePort *port = dynamic_cast< InputImagePort * >( &inputPort );
	if( port ) {
		ConnectOut( *port );
	} else {
		//TODO - throw exception
	}
}

template< typename ElementType, unsigned dimension >
void
OutImageConnection< Image< ElementType, dimension > >
::DisconnectOut( InputPort& inputPort )
{
	InputImagePort *port = dynamic_cast< InputImagePort * >( &inputPort );
	if( port ) {
		DisconnectOut( *port );
	} else {
		//TODO - throw exception
	}
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_CONNECTION_H*/

