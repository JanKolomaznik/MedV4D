#ifndef _CONNECTION_H
#define _CONNECTION_H

#include "Imaging/Image.h"
#include "Imaging/AbstractFilter.h"
#include "Imaging/Ports.h"
#include "Common.h"
#include <map>

namespace M4D
{
namespace Imaging
{

/**
 * Connection object inheriting from interfaces for 
 * input connection and output connection.
 **/
class ConnectionInterface : public MessageRouterInterface
{//TODO - remove two ancestor classes
public:
	class EMismatchPortType;
	class EConnectionOccupied;
	/**
	 * Default constructor
	 **/
	ConnectionInterface() {}

	/**
	 * Virtual destructor.
	 **/
	virtual 
	~ConnectionInterface() {}
	
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

	virtual void 
	DisconnectAll()=0;

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
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ConnectionInterface );
};

class ConnectionInterface::EMismatchPortType
{
public:
	//TODO
};

class ConnectionInterface::EConnectionOccupied
{
public:
	//TODO
};

//******************************************************************************
//******************************************************************************
//******************************************************************************


//We prohibit general usage - only specialized templates used.
template< typename ImageTemplate >
class ImageConnection;


template< typename ElementType, unsigned dimension >
class ImageConnection< Image< ElementType, dimension > >
	: public ConnectionInterface
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > Image;
	typedef typename M4D::Imaging::InputPortImageFilter< Image > InputImagePort;
	typedef typename M4D::Imaging::OutputPortImageFilter< Image > OutputImagePort;
	
	~ImageConnection() {}

	
	void
	ConnectOut( InputPort& inputPort );

	virtual void
	ConnectOutTyped( InputImagePort& inputPort ); 

	void
	ConnectIn( OutputPort& outputPort );

	virtual void
	ConnectInTyped( OutputImagePort& outputPort ); 
	
	void
	DisconnectOut( InputPort& inputPort );

	virtual void
	DisconnectOutTyped( InputImagePort& inputPort ); 

	void
	DisconnectIn();

	void 
	DisconnectAll();

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
	/**
	 * Hidden default constructor - we don't allow direct
	 * construction of object of this class.
	 **/
	ImageConnection() {}

	typename Image::Ptr			_image;
	OutputImagePort				*_input;
	std::map< uint64, InputImagePort* >	_outputs;
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
