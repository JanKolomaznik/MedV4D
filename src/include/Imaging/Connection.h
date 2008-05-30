#ifndef _CONNECTION_H
#define _CONNECTION_H

#include "Imaging/AbstractFilter.h"
#include "Imaging/Ports.h"
#include <boost/shared_ptr.hpp>
#include "Common.h"

namespace M4D
{
namespace Imaging
{

/**
 * Base interface of connection objects. Methods to connect 
 * input ports of filters. !!!See that output from connection 
 * is input to connected filter!!!
 **/
class InConnectionInterface
{
public:
	/**
	 * Default constructor
	 **/
	InConnectionInterface() {}
	
	/**
	 * Virtual destructor.
	 **/
	virtual
	~InConnectionInterface() {}

	/**
	 * Handle input port of some filter. 
	 * !!!Output of this connection!!!
	 * @param inputPort Abstract reference to input port of some filter.
	 **/
	virtual void
	ConnectOut( InputPort& inputPort )=0;

	/**
	 * Disconnect chosen filter if possible.
	 * @param inputPort Port to be disconnected.
	 **/
	virtual void
	DisconnectOut( InputPort& inputPort )=0;

};

/**
 * Base interface of connection objects. Methods to connect 
 * output ports of filters. !!!See that input to connection 
 * is output from connected filter!!!
 **/
class OutConnectionInterface
{
public:
	/**
	 * Default constructor
	 **/
	OutConnectionInterface() {}
	
	/**
	 * Virtual destructor.
	 **/
	virtual
	~OutConnectionInterface() {}

	/**
	 * Handle input port of some filter. 
	 * !!!Output of this connection!!!
	 * @param outputPort Abstract reference to input port of some filter.
	 **/
	virtual void
	ConnectIn( OutputPort& outputPort )=0;

	/**
	 * Disconnect input port. !!!Output port of some filter!!!
	 **/
	virtual void
	DisconnectIn()=0;
};

/**
 * Connection object inheriting from interfaces for 
 * input connection and output connection.
 **/
class Connection : public InConnectionInterface, public OutConnectionInterface
{
public:
	/**
	 * Default constructor
	 **/
	Connection() {}

	/**
	 * Virtual destructor.
	 **/
	virtual 
	~Connection() {}

protected:

private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( Connection );
};

//******************************************************************************
//******************************************************************************
//******************************************************************************

//We prohibit general usage - only specialized templates used.
template<  typename ImageTemplate >
class ImageFilterInConnection;

//We prohibit general usage - only specialized templates used.
template<  typename ImageTemplate >
class ImageFilterOutConnection;


template< typename ElementType, unsigned dimension >
class ImageFilterInConnection< Image< ElementType, dimension > >
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > Image;
	typedef typename M4D::Imaging::InputPortImageFilter< Image > InputImagePort;

	virtual void
	ConnectOut( InputImagePort& inputPort ) = 0; 

	virtual void
	DisconnectOut( InputImagePort& inputPort ) = 0; 

};

template< typename ElementType, unsigned dimension >
class ImageFilterOutConnection< Image< ElementType, dimension > >
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > Image;
	typedef typename M4D::Imaging::OutputPortImageFilter< Image > OutputImagePort;

	virtual void
	ConnectIn( OutputImagePort& outputPort ) = 0; 

};

//We prohibit general usage - only specialized templates used.
template< typename ImageTemplate >
class ImageFilterConnection;


template< typename ElementType, unsigned dimension >
class ImageFilterConnection< Image< ElementType, dimension > >
	: public Connection,
	public ImageFilterInConnection< Image< ElementType, dimension > >,
	public ImageFilterOutConnection< Image< ElementType, dimension > >
{
public:

	ImageFilterConnection() {}

	~ImageFilterConnection() {}

	void
	ConnectIn( OutputPort& outputPort );

	void
	ConnectOut( InputPort& inputPort );

	void
	DisconnectOut( InputPort& inputPort );

protected:

private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageFilterConnection );
};

}/*namespace Imaging*/
}/*namespace M4D*/

//Include implementation
#include "Imaging/Connection.tcc"

#endif /*_CONNECTION_H*/
