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
/*
template< typename ImageTemplate >
class OutImageConnectionSimple;

template< typename ElementType, unsigned dimension >
class OutImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
: public OutImageConnection< M4D::Imaging::Image< ElementType, dimension > >
{
public:
	void
	ConnectOutTyped( typename OutImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		);

	void
	DisconnectOutTyped( typename OutImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		);
	void
	PutOutputImage( M4D::Imaging::Image< ElementType, dimension >::Ptr image );
};

template<  typename ImageTemplate >
class InImageConnectionSimple;

template< typename ElementType, unsigned dimension >
class InImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
: public InImageConnection< M4D::Imaging::Image< ElementType, dimension > >
{
public:
	void
	ConnectInTyped( typename InImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::OutputImagePort& outputPort 
		); 

	void
	DisconnectIn();

	void
	PutInputImage( M4D::Imaging::Image< ElementType, dimension >::Ptr image );

};
*/

template<  typename ImageTemplate >
class ImageConnectionSimple;

template< typename ElementType, unsigned dimension >
class ImageConnectionSimple< M4D::Imaging::Image< ElementType, dimension > >
: public ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
{
public:
	void
	ConnectOutTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		); 

	void
	DisconnectOutTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		);

	void
	ConnectInTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::OutputImagePort& outputPort 
		); 
	
	void
	DisconnectIn();

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
	void
	ConnectOutTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		); 

	void
	DisconnectOutTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		);

	void
	ConnectInTyped( typename ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
			::OutputImagePort& outputPort 
		); 
	
	void
	DisconnectIn();

};


}/*namespace Imaging*/
}/*namespace M4D*/

#include "Imaging/DefaultConnection.tcc"

#endif /*_CONNECTION_H*/
