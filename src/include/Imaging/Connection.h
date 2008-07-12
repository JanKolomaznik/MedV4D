#ifndef _CONNECTION_H
#define _CONNECTION_H

#include "Imaging/Image.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AbstractFilter.h"
#include "Imaging/Ports.h"
#include "Common.h"
#include <map>
#include <algorithm>

namespace M4D
{
namespace Imaging
{

/**
 * Connection object inheriting from interfaces for 
 * input connection and output connection.
 **/
class ConnectionInterface : public MessageRouterInterface
{
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
	ConnectConsumer( InputPort& inputPort )=0;

	/**
	 * Disconnect chosen filter if possible.
	 * @param inputPort Port to be disconnected.
	 **/
	virtual void
	DisconnectConsumer( InputPort& inputPort )=0;

	/**
	 * Handle input port of some filter. 
	 * !!!Output of this connection!!!
	 * @param outputPort Abstract reference to input port of some filter.
	 **/
	virtual void
	ConnectProducer( OutputPort& outputPort )=0;

	/**
	 * Disconnect input port. !!!Output port of some filter!!!
	 **/
	virtual void
	DisconnectProducer()=0;

	virtual void 
	DisconnectAll()=0;

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		){ /*empty*/ }


	/**
	 * Method, which try to lock dataset controled by this connection for 
	 * reading or editing information contained in it - this lock synchronize only
	 * with structure editing. 
	 * Only situation when this method fail (return false) is, when dataset
	 * is locked for structure edit.
	 * \return False if unsuccessfull.
	 **/
	virtual bool 
	TryLockDataset() = 0;

	/**
	 * Method which remove one lock from dataset.
	 **/
	virtual void
	UnlockDataset() = 0;

	/**
	 * Method try to lock dataset exclusively - TryLockDataset() will fail.
	 * \return False if unsuccessfull.
	 **/
	virtual bool
	TryExclusiveLockDataset() = 0;

	virtual void
	ExclusiveUnlockDataset() = 0;

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

/**
 * Not supposed to instantiate - use only as substitution for typed images connections.
 **/
class AbstractImageConnection : public ConnectionInterface
{
public:

	virtual const AbstractImage &
	GetAbstractImageReadOnly()const = 0;

	/*virtual void
	ConnectAImageConsumer( InputPortAbstractImage &inputPort );*/
protected:
	typedef std::map< uint64, InputPortAbstractImage* > ConsumersMap;

	void
	PushConsumer( InputPortAbstractImage& consumer );
	
	ConsumersMap				_consumers;
};

//We prohibit general usage - only specialized templates used.
template< typename ImageTemplate >
class ImageConnection;


template< typename ElementType, unsigned dimension >
class ImageConnection< Image< ElementType, dimension > >
	: public AbstractImageConnection
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > Image;
	typedef typename M4D::Imaging::InputPortImageFilter< Image > InputImagePort;
	typedef typename M4D::Imaging::OutputPortImageFilter< Image > OutputImagePort;
	
	~ImageConnection() {}

	
	void
	ConnectConsumer( InputPort& inputPort );

	/*virtual void
	ConnectConsumerTyped( InputImagePort& inputPort ); */

	void
	ConnectProducer( OutputPort& outputPort );

	/*virtual void
	ConnectProducerTyped( OutputImagePort& outputPort ); */
	
	void
	DisconnectConsumer( InputPort& inputPort );

	virtual void
	DisconnectConsumerTyped( InputImagePort& inputPort ); 

	void
	DisconnectProducer();

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

	const AbstractImage &
	GetAbstractImageReadOnly()const
		{
			return GetImageReadOnly();
		}

	void
	SetImageSize( 
			size_t 		minimums[ dimension ], 
			size_t 		maximums[ dimension ], 
			float32		elementExtents[ dimension ]
		    );

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
	
	ImageConnection( typename Image::Ptr image ) 
		: _image( image ) {}

	typename Image::Ptr			_image;
	OutputImagePort				*_input;
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
