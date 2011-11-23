/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConnectionInterface.h 
 * @{ 
 **/

#ifndef _CONNECTION_INTERFACE_H
#define _CONNECTION_INTERFACE_H

#include "common/Common.h"
//#include "Imaging/Ports.h"
#include "Imaging/PipelineMessages.h"
#include "Imaging/ADataset.h"

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

template< typename DatasetType >
class OutputPortTyped;

template< typename DatasetType >
class InputPortTyped;


/**
 * Connection object inheriting from interfaces for 
 * input connection and output connection.
 **/
class ConnectionInterface : public MessageRouterInterface
{
public:
	class EMismatchPortType;
	class EConnectionOccupied;
	class ENoDatasetAssociated;

	/**
	 * Default constructor
	 **/
	ConnectionInterface():	_producer( NULL ) {}

	/**
	 * Virtual destructor.
	 **/
	virtual 
	~ConnectionInterface();
	
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
	void
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
	void
	DisconnectProducer();
	
	/**
	 * Disconnect all ports.
	 **/
	void 
	DisconnectAll();

	/**
	 * Method used to put dataset to connection. 
	 * Connection gets ownership over dataset - old one will be released
	 * \param dataset Smart pointer to dataset - must be valid.
	 **/
	virtual void
	PutDataset( ADataset::Ptr dataset )=0;

	void
	ResetDataset()
		{ PutDataset( ADataset::Ptr() ); }
	/**
	 * \return Reference to dataset under control.
	 **/
	virtual ADataset &
	GetDataset()const = 0;

	virtual ADataset::Ptr
	GetDatasetPtr()const = 0;

	/**
	 * \return Constant reference to dataset under control.
	 **/
	virtual const ADataset &
	GetDatasetReadOnly()const = 0;

	virtual ADataset::ConstPtr
	GetDatasetReadOnlyPtr()const = 0;


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

class ConnectionInterface::ENoDatasetAssociated
{
public:
	//TODO
};

//******************************************************************************
template< typename DatasetType >
class ConnectionInterfaceTyped: public ConnectionInterfaceTyped< typename DatasetType::PredecessorType >
{
public:
	typedef	ConnectionInterfaceTyped< DatasetType > ThisClass;
	typedef ConnectionInterfaceTyped< typename DatasetType::PredecessorType >	PredecessorType;
	typedef OutputPortTyped< DatasetType >	ProducerPortType;
	typedef InputPortTyped< DatasetType >	ConsumerPortType;

	void
	ConnectConsumer( InputPort& inputPort );

	void
	ConnectProducer( OutputPort& outputPort );

	DatasetType &
	GetDatasetTyped()const
		{ return DatasetType::Cast( this->GetDataset() ); }

	typename DatasetType::Ptr
	GetDatasetTypedPtr()const
		{ return DatasetType::Cast( this->GetDatasetPtr() ); }

	const DatasetType &
	GetDatasetReadOnlyTyped()const
		{ return DatasetType::Cast( this->GetDatasetReadOnly() ); }

	typename DatasetType::ConstPtr
	GetDatasetReadOnlyTypedPtr()const
		{ return DatasetType::Cast( this->GetDatasetReadOnlyPtr() ); }

protected:
};

template<>
class ConnectionInterfaceTyped< ADataset >: public ConnectionInterface
{
public:
	typedef InputPortTyped< ADataset > ConsumerPortType;

protected:
};

template< typename DatasetType >
class ConnectionTyped: public ConnectionInterfaceTyped< DatasetType >
{
public:
	ConnectionTyped( bool ownsDataset = true );

	void
	PutDataset( ADataset::Ptr dataset )
		{
			typename DatasetType::Ptr newDataset;
			if( dataset ) {
				newDataset = DatasetType::Cast( dataset );
				
			}

			if( _dataset ) {
				_dataset->ExclusiveLockDataset();
				typename DatasetType::Ptr tmp = _dataset;
				_dataset = newDataset;
				tmp->ExclusiveUnlockDataset();
			} else {
				_dataset = newDataset;
			}
			

			this->RouteMessage( 
				MsgDatasetPut::CreateMsg(), 
				PipelineMessage::MSS_NORMAL,
				FD_BOTH	
			);
		}

	ADataset &
	GetDataset()const
		{ if( !_dataset ) { _THROW_ ConnectionInterface::ENoDatasetAssociated(); }
			return *_dataset;
		}

	ADataset::Ptr
	GetDatasetPtr()const
		{ if( !_dataset ) { _THROW_ ConnectionInterface::ENoDatasetAssociated(); }
			return boost::static_pointer_cast< ADataset >( _dataset );
		}

	const ADataset &
	GetDatasetReadOnly()const
		{ if( !_dataset ) { _THROW_ ConnectionInterface::ENoDatasetAssociated(); }
			return *_dataset;
		}

	ADataset::ConstPtr
	GetDatasetReadOnlyPtr()const
		{ if( !_dataset ) { _THROW_ ConnectionInterface::ENoDatasetAssociated(); }
			return boost::static_pointer_cast< const ADataset >( _dataset );
		}
protected:
	mutable typename DatasetType::Ptr _dataset;
};

//******************************************************************************


}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

//Include implementation
#include "Imaging/ConnectionInterface.tcc"

#endif /*_CONNECTION_INTERFACE_H*/

/** @} */

