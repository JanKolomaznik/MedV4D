#ifndef _DEFAULT_CONNECTION_H
#define _DEFAULT_CONNECTION_H

#include "Imaging/AbstractFilter.h"
#include "Imaging/Ports.h"
#include "Imaging/Connection.h"
#include "Common.h"


namespace M4D
{
namespace Imaging
{


template<  typename ImageTemplate >
class ImageConnectionSimple;

template< typename ElementType, unsigned dimension >
class ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
: public ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
{
public:

	void
	PutImage( typename M4D::Imaging::Image< ElementType, dimension >::Ptr image );
};

template<  typename ImageTemplate >
class ImageConnectionImOwner;

template< typename ElementType, unsigned dimension >
class ImageConnectionImOwner< M4D::Imaging::Image< ElementType, dimension > >
: public ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
{
public:
	ImageConnectionImOwner();
	//TODO

};


}/*namespace Imaging*/
}/*namespace M4D*/

#include "Imaging/DefaultConnection.tcc"

#endif /*_CONNECTION_H*/
