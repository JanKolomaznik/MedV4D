#ifndef _CONNECTION_H
#define _CONNECTION_H

#include "Imaging/AbstractFilter.h"
#include "Imaging/Ports.h"
#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace Imaging
{

class Connection
{
public:
	//typedef boost::shared_ptr< Connection > ConnectionPtr;

	Connection() {}

	virtual 
	~Connection() {}

	virtual void
	ConnectIn( OutputPort& outputPort )=0;

	virtual void
	ConnectOut( InputPort& inputPort )=0;

	virtual void
	DisconnectIn()=0;

	virtual void
	DisconnectOut( InputPort& inputPort )=0;

protected:

private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( Connection );
};

//We prohibit general usage - only specialized templates used.
template< typename ImageTemplate >
class ImageFilterConnection;


template< typename ElementType, unsigned dimension >
class ImageFilterConnection< Image< ElementType, dimension > >: public Connection
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > Image;
	typedef typename M4D::Imaging::InputPortImageFilter< Image > InputImagePort;
	typedef typename M4D::Imaging::OutputPortImageFilter< Image > OutputImagePort;

	ImageFilterConnection() {}

	~ImageFilterConnection() {}

	void
	ConnectIn( OutputPort& outputPort );

	virtual void
	ConnectIn( OutputImagePort& outputPort ) = 0; 

	void
	ConnectOut( InputPort& inputPort );

	virtual void
	ConnectOut( InputImagePort& inputPort ) = 0; 

	virtual void
	DisconnectIn()=0;

	void
	DisconnectOut( InputPort& inputPort );

	virtual void
	DisconnectOut( InputImagePort& inputPort ) = 0; 
protected:

private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageFilterConnection );
};

//Include implementation
#include "Imaging/Connection.tcc"

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_CONNECTION_H*/
