#ifndef _CONNECTION_INTERFACE_H
#define _CONNECTION_INTERFACE_H

#include "Common.h"
//#include "Imaging/Ports.h"
#include "Imaging/PipelineMessages.h"
#include "Imaging/AbstractDataSet.h"

#include <map>
#include <algorithm>

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

	virtual void 
	DisconnectAll();

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		){ /*empty*/ }

	virtual AbstractDataSet &
	GetDataset()const = 0;

	virtual const AbstractDataSet &
	GetDatasetReadOnly()const = 0;

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
	
	ConsumersMap				_consumers;

	OutputPort				*_producer;
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


#endif /*_CONNECTION_INTERFACE_H*/
