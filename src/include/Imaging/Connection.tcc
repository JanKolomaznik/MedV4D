#ifndef _CONNECTION_H
#error File Connection.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename ElementType, unsigned dimension >
void
ImageFilterConnection< Image< ElementType, dimension > >
::ConnectIn( OutputPort& outputPort )
{

}

template< typename ElementType, unsigned dimension >
void
ImageFilterConnection< Image< ElementType, dimension > >
::ConnectOut( InputPort& inputPort )
{

}

template< typename ElementType, unsigned dimension >
void
ImageFilterConnection< Image< ElementType, dimension > >
::DisconnectOut( InputPort& inputPort )
{

}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_CONNECTION_H*/

