/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Ports.h 
 * @{ 
 **/

#ifndef _PORTS_H
#define _PORTS_H

#include <boost/shared_ptr.hpp>
#include "Imaging/ADataset.h"
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

template< typename DatasetType >
class ConnectionInterfaceTyped;

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



	virtual ConnectionInterface*
	CreateIdealConnectionObject( bool ownsDataset ) = 0;

	virtual bool
	IsConnectionCompatible( ConnectionInterface &conn ) = 0;

	virtual unsigned
	GetHierarchyDepth()const = 0;
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

class Port::EConnectionTypeMismatch: public ErrorHandling::ExceptionBase
{
public:
	EConnectionTypeMismatch() throw() : ExceptionBase( "Port type mismatch." ){}
	~EConnectionTypeMismatch() throw(){}

	//TODO
};
/**
 * Exception which is thrown in situations, when port isn't
 * connected and method need port to be connected for succesful
 * execution.
 **/
class Port::EDisconnected: public ErrorHandling::ExceptionBase
{
public:
	EDisconnected( uint64 port ) throw() : ExceptionBase( "Port Disconnected." ), _port( port ) {}
	EDisconnected( const Port &port ) throw() : ExceptionBase( "Port Disconnected." ), _port( port._id ) {}
	~EDisconnected() throw(){}
	//TODO
protected:
	/**
	 * ID of port which caused this exception.
	 **/
	uint64  _port;
};

class Port::EPortAlreadyConnected: public ErrorHandling::ExceptionBase
{
public:
	EPortAlreadyConnected() throw() : ExceptionBase( "Port already connected." ){}
	~EPortAlreadyConnected() throw(){}

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

	virtual	const ADataset&
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
	
	virtual	ADataset&
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

//******************************************************************************
template< typename DatasetType >
class InputPortTyped: public InputPortTyped< typename DatasetType::PredecessorType >
{
public:
	typedef ConnectionInterfaceTyped< DatasetType > IdealConnectionInterface;
	typedef InputPortTyped< typename DatasetType::PredecessorType > PredecessorType;
	static const unsigned HierarchyDepth = DatasetType::HierarchyDepth;

	InputPortTyped() {}

	const DatasetType&
	GetDatasetTyped()const;

	const ADataset&
	GetDataset()const
		{ return GetDatasetTyped(); }

	void
	Plug( ConnectionInterface & connection );

	ConnectionInterface*
	CreateIdealConnectionObject( bool ownsDataset );
	
	bool
	IsConnectionCompatible( ConnectionInterface &conn );
protected:
	
};

template<>
class InputPortTyped< ADataset >: public InputPort
{
public:
	typedef ConnectionInterfaceTyped< ADataset > IdealConnectionInterface;
	typedef InputPort PredecessorType;
	static const unsigned HierarchyDepth = ADataset::HierarchyDepth;

	unsigned
	GetHierarchyDepth()const
		{ return HierarchyDepth; }
	bool
	IsConnectionCompatible( ConnectionInterface &conn );
};

//******************************************************************************
template< typename DatasetType >
class OutputPortTyped: public OutputPortTyped< typename DatasetType::PredecessorType >
{
public:
	typedef ConnectionInterfaceTyped< DatasetType > IdealConnectionInterface;
	typedef OutputPortTyped< typename DatasetType::PredecessorType > PredecessorType;
	static const unsigned HierarchyDepth = DatasetType::HierarchyDepth;

	OutputPortTyped() {}

	DatasetType&
	GetDatasetTyped()const;

	ADataset&
	GetDataset()const
		{ return GetDatasetTyped(); }

	void
	Plug( ConnectionInterface & connection );

	
	ConnectionInterface*
	CreateIdealConnectionObject( bool ownsDataset );

	unsigned
	GetHierarchyDepth()const
		{ return HierarchyDepth; }
	bool
	IsConnectionCompatible( ConnectionInterface &conn );
protected:

};

template<>
class OutputPortTyped< ADataset >: public OutputPort
{
public:
	typedef ConnectionInterfaceTyped< ADataset > IdealConnectionInterface;
	typedef OutputPort PredecessorType;
	static const unsigned HierarchyDepth = ADataset::HierarchyDepth;

	unsigned
	GetHierarchyDepth()const
		{ return HierarchyDepth; }
	bool
	IsConnectionCompatible( ConnectionInterface &conn );
};
//******************************************************************************

class PortList: public MessageSenderInterface
{

public:
	class EWrongPortIndex;

	PortList(): _size( 0 ) {}


	size_t
	Size() const
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
	AppendPort( InputPort* port );

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
	AppendPort( OutputPort* port );

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

class PortList::EWrongPortIndex: public ErrorHandling::EBadIndex
{
public:
	EWrongPortIndex( int32 idx ): ErrorHandling::EBadIndex( "Accessing port in portlist by wrong index", idx )
	{}
};


//******************************************************************************

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#include "Imaging/Ports.tcc"

#endif /*_PORTS_H*/


/** @} */

