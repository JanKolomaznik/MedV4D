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
class OutImageConnectionSimple;

template< typename ElementType, unsigned dimension >
class OutImageConnectionSimple< Image< ElementType, dimension > >
: public OutImageConnection< Image< ElementType, dimension > >
{
public:
	void
	ConnectOut( typename OutImageConnection< Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		);

	void
	DisconnectOut( typename OutImageConnection< Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		);
};

template<  typename ImageTemplate >
class InImageConnectionSimple;

template< typename ElementType, unsigned dimension >
class InImageConnectionSimple< Image< ElementType, dimension > >
: public InImageConnection< Image< ElementType, dimension > >
{
public:
	void
	ConnectIn( typename InImageConnection< Image< ElementType, dimension > >
			::OutputImagePort& outputPort 
		); 

	void
	DisconnectIn();

};

template<  typename ImageTemplate >
class ImageConnectionSimple;

template< typename ElementType, unsigned dimension >
class ImageConnectionSimple< Image< ElementType, dimension > >
: public ImageConnection< Image< ElementType, dimension > >
{
public:
	void
	ConnectOut( typename OutImageConnection< Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		); 

	void
	DisconnectOut( typename OutImageConnection< Image< ElementType, dimension > >
			::InputImagePort& inputPort 
		);

	void
	ConnectIn( typename InImageConnection< Image< ElementType, dimension > >
			::OutputImagePort& outputPort 
		); 
	
	void
	DisconnectIn();
};


}/*namespace Imaging*/
}/*namespace M4D*/

#include "Imaging/DefaultConnection.tcc"

#endif /*_CONNECTION_H*/
