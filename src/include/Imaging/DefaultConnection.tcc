#ifndef _DEFAULT_CONNECTION_H
#error File DefaultConnection.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{
/*
template< typename ElementType, unsigned dimension >
void
OutImageConnectionSimple< Image< ElementType, dimension > >
::ConnectOutTyped( typename OutImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
OutImageConnectionSimple< Image< ElementType, dimension > >
::DisconnectOutTyped( typename OutImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

}
*/
//******************************************************************************
/*
template< typename ElementType, unsigned dimension >
void
InImageConnectionSimple< Image< ElementType, dimension > >
::ConnectInTyped( typename InImageConnection< M4D::Imaging::Image< ElementType, dimension > >
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
*/
//******************************************************************************

template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::ConnectOutTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectOutTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
		::InputImagePort& inputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::ConnectInTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::OutputImagePort &outputPort 
	)
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::DisconnectIn()
{

}

template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::PutImage( typename M4D::Imaging::Image< ElementType, dimension >::Ptr image )
{

}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_DEFAULT_CONNECTION_H*/

