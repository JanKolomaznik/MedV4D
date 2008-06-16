#ifndef _DEFAULT_CONNECTION_H
#error File DefaultConnection.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

//******************************************************************************


template< typename ElementType, unsigned dimension >
void
ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
::PutImage( typename M4D::Imaging::Image< ElementType, dimension >::Ptr image )
{
	//TODO exception
	this->_image = image;
}

template< typename ElementType, unsigned dimension >
ImageConnectionImOwner< M4D::Imaging::Image< ElementType, dimension > >
::ImageConnectionImOwner()
{
	this->_image = typename M4D::Imaging::Image< ElementType, dimension >::Ptr( 
			new M4D::Imaging::Image< ElementType, dimension >() 
			);
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_DEFAULT_CONNECTION_H*/

