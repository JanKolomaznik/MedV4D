/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Ports.h 
 * @{ 
 **/

#ifndef _PORTS_H
#define _PORTS_H

#include <boost/shared_ptr.hpp>
#include "Imaging/AbstractDataSet.h"
#include <vector>
#include "Imaging/PipelineMessages.h"
//#include "Imaging/ConnectionInterface.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

//Forward declarations *****************
class ConnectionInterface;


//**************************************

class Port: public MessageOperatorInterface
{
public:

	/**
	 * Exception which is thrown in situations, when port isn't
	 * connected and method need port to be connected for succesful
	 * execution.
	 **/
	class EDisconnected;
	class EConnectionTypeMismatch;
	class EPortAlreadyConnected;

	/**
	 * Default constructor - port obtain unique ID.
	 **/
	Port(): _connection( NULL )
		{ _id = GenerateUniqueID(); }

	/**
	 * Virtual destructor - class can be (and will be) disposed polymorphicaly.
	 **/
	virtual
	~Port() { }

	/**
	 * return True if port is plugged to connection object.
	 **/
	bool
	IsPlugged()const
		{ return _connection != NULL; }
	
	virtual void
	Plug( ConnectionInterface & connection ) = 0;	

	ConnectionInterface *
	GetConnection()const
		{ return _connection; }

	/**
	 * Method to unplug port from connection object - if already 
	 * disconnected do nothing.
	 **/
	virtual void
	UnPlug( bool onlyYourself = false ) = 0;

	uint64
	GetID()const
		{ return _id; }	

	MessageReceiverInterface *
	GetReceiver()
		{ return _msgReceiver; }

	void
	SetReceiver( MessageReceiverInterface	*msgReceiver )
		{ _msgReceiver = msgReceiver; }


	void
	ReceiveMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle, 
		FlowDirection				direction
		);

	/**
	 * Method called when someone wants to obtain dataset, accessible
	 * through port. It doesn't actualy lock data stored inside dataset,
	 * only disallow to change internal structure - release or reallocate 
	 * buffer, etc.
	 * \return True if lock was successful, false otherwise.
	 **/
	virtual bool 
	TryLockDataset();

	virtual void
	LockDataset();

	//TODO - if store, wheather this port already locked dataset - unlock during destruction ...
	/**
	 * Release dataset lock, which was locked by previous method.
	 **/
	void 
	ReleaseDatasetLock();
					
protected:


	ConnectionInterface *_connection;
private:
	/**
	 * Method for unique port ID generation - thread safe.
	 **/
	static uint64
	GenerateUniqueID();

	uint64	_id;
	/**
	 * Pointer to object which will obtain messages received by port.
	 **/
	MessageReceiverInterface	*_msgReceiver;	
};

class Port::EConnectionTypeMismatch
{
public:
	//TODO
};
/**
 * Exception which is thrown in situations, when port isn't
 * connected and method need port to be connected for succesful
 * execution.
 **/
class Port::EDisconnected
{
public:
	EDisconnected( uint64 port ) : _port( port ) {}
	//TODO
protected:
	/**
	 * ID of port which caused this exception.
	 **/
	uint64  _port;
};

class Port::EPortAlreadyConnected
{
public:
	//TODO
};

class InputPort: public Port
{
public:
	virtual
	~InputPort() {}

	/**
	 * Method to unplug port from connection object - if already 
	 * disconnected do nothing.
	 **/
	void
	UnPlug( bool onlyYourself = false );

	virtual	const AbstractDataSet&
	GetDataset()const = 0;

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		);
	
protected:
	void
	PortPluggedMsg()
		{ ReceiveMessage( MsgPortPlugged::CreateMsg(), 
			PipelineMessage::MSS_NORMAL,
			FD_IN_FLOW );
		};
private:

};

class OutputPort: public Port
{
public:
	virtual
	~OutputPort() {}

	/**
	 * Method to unplug port from connection object - if already 
	 * disconnected do nothing.
	 **/
	void
	UnPlug( bool onlyYourself = false );
	
	virtual	AbstractDataSet&
	GetDataset()const = 0;

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		);
protected:
	void
	PortPluggedMsg()
		{ ReceiveMessage( MsgPortPlugged::CreateMsg(), 
			PipelineMessage::MSS_NORMAL,
			FD_AGAINST_FLOW	);
		};

private:

};

template< typename DatasetType >
class InputPortTyped: public InputPortTyped< typename DatasetType::PredecessorType >
{
public:
	InputPortTyped() {}

	const DatasetType&
	GetDatasetTyped()const;

	const AbstractDataSet&
	GetDataset()const
		{ return GetDatasetTyped(); }

	void
	Plug( ConnectionInterface & connection );

protected:
	
};

template< typename DatasetType >
class OutputPortTyped: public OutputPortTyped< typename DatasetType::PredecessorType >
{
public:
	OutputPortTyped() {}

	DatasetType&
	GetDatasetTyped()const;

	AbstractDataSet&
	GetDataset()const
		{ return GetDatasetTyped(); }

	void
	Plug( ConnectionInterface & connection );

protected:

};

//******************************************************************************

class PortList: public MessageSenderInterface
{

public:
	PortList(): _size( 0 ) {}


	size_t
	Size()
	{ return _size; }
protected:
	virtual ~PortList() {}

	size_t	_size;
private:
	//Not implemented
	PortList( const PortList& );
	PortList&
	operator=( const PortList& );
};

class InputPortList: public PortList
{
public:
	InputPortList( MessageReceiverInterface *msgReceiver ) 
		: _msgReceiver( msgReceiver )
       		{ /*TODO check pointer*/ }

	~InputPortList();

	size_t
	AddPort( InputPort* port );

	InputPort &
	GetPort( size_t idx )const;
	
	template< typename PortType >
	PortType&
	GetPortTyped( size_t idx )const;

	template< typename PortType >
	PortType*
	GetPortTypedSafe( size_t idx )const;

	InputPort &
	operator[]( size_t idx )const
	{ return GetPort( idx ); }

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle
		);
private:
	//Not implemented
	InputPortList( const InputPortList& );
	InputPortList();
	InputPortList&
	operator=( const InputPortList& );


	std::vector< InputPort* >	_ports;

	MessageReceiverInterface	*_msgReceiver;	
};

class OutputPortList: public PortList
{
public:
	OutputPortList( MessageReceiverInterface *msgReceiver ) 
		: _msgReceiver( msgReceiver )
       		{ /*TODO check pointer*/ }

	~OutputPortList();

	size_t
	AddPort( OutputPort* port );

	OutputPort &
	GetPort( size_t idx )const;
	
	template< typename PortType >
	PortType&
	GetPortTyped( size_t idx )const;

	template< typename PortType >
	PortType*
	GetPortTypedSafe( size_t idx )const;

	OutputPort &
	operator[]( size_t idx )const
	{ return GetPort( idx ); }

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle
		);
private:
	//Not implemented
	OutputPortList( const OutputPortList& );
	OutputPortList();
	OutputPortList&
	operator=( const OutputPortList& );


	std::vector< OutputPort* >	_ports;

	MessageReceiverInterface	*_msgReceiver;	
};


//******************************************************************************

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#include "Imaging/Ports.tcc"

#endif /*_PORTS_H*/


/** @} */

