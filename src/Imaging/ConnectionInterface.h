#ifndef _CONNECTION_INTERFACE_H
#define _CONNECTION_INTERFACE_H

#include "Common.h"
//#include "Imaging/Ports.h"
#include "Imaging/PipelineMessages.h"
#include "Imaging/AbstractDataSet.h"

#include <map>
#include <algorithm>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

class InputPort;
class OutputPort;


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
	ConnectionInterface():	_producer( NULL ) {}

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
	DisconnectConsumer( InputPort& inputPort );

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
	DisconnectProducer();
	
	/**
	 * Disconnect all ports.
	 **/
	virtual void 
	DisconnectAll();

	/**
	 * Method used to put dataset to connection. 
	 * Connection gets ownership over dataset - old one will be released
	 * \param dataset Smart pointer to dataset - must be valid.
	 **/
	virtual void
	PutDataset( AbstractDataSet::ADataSetPtr dataset )=0;

	/**
	 * \return Reference to dataset under control.
	 **/
	virtual AbstractDataSet &
	GetDataset()const = 0;

	/**
	 * \return Constant reference to dataset under control.
	 **/
	virtual const AbstractDataSet &
	GetDatasetReadOnly()const = 0;

	/**
	 * Sets object, which will be listening messages going through
	 * connection. Other listeners are untouched and gets messages too.
	 **/
	void
	SetMessageHook( MessageReceiverInterface::Ptr hook )
		{ _messageHook = hook; }

	/**
	 * This method will resend message according to other parameters.
	 * \param msg Smart pointer to message structure.
	 * \param sendStyle How the message should be send (broadcast, etc.)
	 * \param direction If mesage goes in flow or against flow of pipeline.
	 **/
	void
	RouteMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle, 
		FlowDirection				direction
		);

	/**
	 * Method, which try to lock dataset controled by this connection for 
	 * reading or editing information contained in it - this lock synchronize only
	 * with structure editing. 
	 * Only situation when this method fail (return false) is, when dataset
	 * is locked for structure edit.
	 * \return False if unsuccessfull.
	 **/
	//virtual bool 
	//TryLockDataset() = 0;

	/**
	 * Method which remove one lock from dataset.
	 **/
	//virtual void
	//UnlockDataset() = 0;

	/**
	 * Method try to lock dataset exclusively - TryLockDataset() will fail.
	 * \return False if unsuccessfull.
	 **/
	//virtual bool
	//TryExclusiveLockDataset() = 0;

	//virtual void
	//ExclusiveUnlockDataset() = 0;

protected:
	typedef std::map< uint64, InputPort* > ConsumersMap;

	void
	PushConsumer( InputPort& consumer );

	/**
	 * Container for all consumers ( input ports ).
	 **/	
	ConsumersMap				_consumers;

	/**
	 * Single producer - output port.
	 **/
	OutputPort				*_producer;

	MessageReceiverInterface::Ptr		_messageHook;
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

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_CONNECTION_INTERFACE_H*/
