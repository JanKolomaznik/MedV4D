#ifndef _DEFAULT_CONNECTION_H
#error File DefaultConnection.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename ElementType, unsigned dimension >
void
OutImageConnectionSimple< Image< ElementType, dimension > >
::ConnectOut( typename OutImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
OutImageConnectionSimple< Image< ElementType, dimension > >
::DisconnectOut( typename OutImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

}

//******************************************************************************

template< typename ElementType, unsigned dimension >
void
InImageConnectionSimple< Image< ElementType, dimension > >
::ConnectIn( typename InImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::OutputImagePort &outputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
InImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectIn()
{

}

//******************************************************************************

template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::ConnectOut( typename OutImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectOut( typename OutImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::ConnectIn( typename InImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::OutputImagePort &outputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >::DisconnectIn()
{

}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_DEFAULT_CONNECTION_H*/

