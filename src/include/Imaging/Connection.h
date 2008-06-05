#ifndef _CONNECTION_H
#define _CONNECTION_H

#include "Imaging/Image.h"
#include "Imaging/AbstractFilter.h"
#include "Imaging/Ports.h"
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
class Connection : public InConnectionInterface, public OutConnectionInterface, 
	public MessageRouterInterface
{//TODO - remove two ancestor classes
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

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		){ /*empty*/ }

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
/*
//We prohibit general usage - only specialized templates used.
template<  typename ImageTemplate >
class InImageConnection;

//We prohibit general usage - only specialized templates used.
template<  typename ImageTemplate >
class OutImageConnection;*/


/*template< typename ElementType, unsigned dimension >
class OutImageConnection< Image< ElementType, dimension > >
	: public OutConnectionInterface
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > Image;
	typedef typename M4D::Imaging::InputPortImageFilter< Image > InputImagePort;

	void
	ConnectOut( InputPort& inputPort );

	virtual void
	ConnectOutTyped( InputImagePort& inputPort ) = 0; 

	void
	DisconnectOut( InputPort& inputPort );

	virtual void
	DisconnectOutTyped( InputImagePort& inputPort ) = 0; 
};

template< typename ElementType, unsigned dimension >
class InImageConnection< Image< ElementType, dimension > >
	: public InConnectionInterface
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > Image;
	typedef typename M4D::Imaging::OutputPortImageFilter< Image > OutputImagePort;

	void
	ConnectIn( OutputPort& outputPort );

	virtual void
	ConnectInTyped( OutputImagePort& outputPort ) = 0; 

};
*/

//We prohibit general usage - only specialized templates used.
template< typename ImageTemplate >
class ImageConnection;


template< typename ElementType, unsigned dimension >
class ImageConnection< Image< ElementType, dimension > >
	: public Connection
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > Image;
	typedef typename M4D::Imaging::InputPortImageFilter< Image > InputImagePort;
	typedef typename M4D::Imaging::OutputPortImageFilter< Image > OutputImagePort;

	


	ImageConnection() {}

	~ImageConnection() {}

	
	void
	ConnectOut( InputPort& inputPort );

	virtual void
	ConnectOutTyped( InputImagePort& inputPort ) = 0; 

	void
	DisconnectOut( InputPort& inputPort );

	virtual void
	DisconnectOutTyped( InputImagePort& inputPort ) = 0; 
	

	void
	ConnectIn( OutputPort& outputPort );

	virtual void
	ConnectInTyped( OutputImagePort& outputPort ) = 0; 

	Image &
	GetImage() 
		{ if( !_image ) { throw ENoImageAssociated(); }
			return *_image;
		}

	const Image &
	GetImageReadOnly()const
		{ if( !_image ) { throw ENoImageAssociated(); }
			return *_image;
		}
	
	void
	RouteMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle,
		FlowDirection				direction
		);

protected:

	 typename Image::Ptr	_image;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageConnection );

public:
	/**
	 * Exception thrown when requiring image object and none 
	 * is available.
	 **/
	class ENoImageAssociated
	{
		//TODO
	};
};



}/*namespace Imaging*/
}/*namespace M4D*/

//Include implementation
#include "Imaging/Connection.tcc"

#endif /*_CONNECTION_H*/
