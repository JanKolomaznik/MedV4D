#ifndef _PORTS_H
#define _PORTS_H

#include <boost/shared_ptr.hpp>
#include "Imaging/AbstractDataSet.h"
#include "Imaging/Image.h"
#include <vector>
#include "Imaging/PipelineMessages.h"

namespace M4D
{
namespace Imaging
{

//Forward declarations *****************
class ConnectionInterface;
class AbstractImageConnection;

template< typename ImageTemplate >
class ImageConnection;

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

	/**
	 * Default constructor - port obtain unique ID.
	 **/
	Port() { _id = GenerateUniqueID(); }

	/**
	 * Virtual destructor - class can be (and will be) disposed polymorphicaly.
	 **/
	virtual
	~Port() { }

	/**
	 * return True if port is plugged to connection object.
	 **/
	virtual	bool
	IsPlugged()const = 0;
	
	virtual void
	Plug( ConnectionInterface & connection ) = 0;	
	/**
	 * Method to unplug port from connection object - if already 
	 * disconnected do nothing.
	 **/
	virtual void
	UnPlug() = 0;

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
protected:
	/**
	 * Method for unique port ID generation - thread safe.
	 **/

private:
	static uint64
	GenerateUniqueID();

	uint64	_id;

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



class InputPort: public Port
{
public:
	virtual
	~InputPort() {}

	/**
	 * Method called when someone wants to obtain dataset, accessible
	 * through port. It doesn't actualy lock data stored inside dataset,
	 * only disallow to change internal structure - release or reallocate 
	 * buffer, etc.
	 * \return True if lock was successful, false otherwise.
	 **/
	virtual bool 
	GetDatasetReadLock();

	//TODO - if store, wheather this port already locked dataset - unlock during destruction ...
	/**
	 * Release dataset lock, which was locked by previous method.
	 **/
	virtual void 
	ReleaseDatasetLock();
	
protected:

private:

};

class OutputPort: public Port
{
public:
	virtual
	~OutputPort() {}

	virtual bool 
	GetDatasetWriteLock();

	//TODO - if store, wheather this port already locked dataset - unlock during destruction ...
	/**
	 * Release dataset lock, which was locked by previous method.
	 **/
	virtual void 
	ReleaseDatasetLock();
protected:

private:

};

//******************************************************************************
class InputPortAbstractImage: public InputPort
{
public:
	const AbstractImage&
	GetAbstractImage()const;

	void
	Plug( ConnectionInterface & connection );

	/*void
	PlugTyped( AbstractImageConnection & connection );*/

	void
	UnPlug();

	bool
	IsPlugged()const
		{ return _abstractImageConnection != NULL; }

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		);
protected:
	AbstractImageConnection	*_abstractImageConnection;

};

/*
class OutputPortAbstractImage: public OutputPort
{
public:
	AbstractImage&
	GetImage()const;
};*/
//******************************************************************************
template< typename ImageType >
class InputPortImageFilter;

template< typename ElementType, unsigned dimension >
class InputPortImageFilter< Image< ElementType, dimension > >: public InputPortAbstractImage
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > ImageType;

	InputPortImageFilter(): _imageConnection( NULL ) {}

	const ImageType&
	GetImage()const;
	
	void
	Plug( ConnectionInterface & connection );

	/*void
	PlugTyped( ImageConnection< ImageType > & connection );*/

	void
	UnPlug();

	bool
	IsPlugged()const
		{ return _imageConnection != NULL; }

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		);


protected:
	
	ImageConnection< ImageType >	*_imageConnection;
};

template< typename ImageType >
class OutputPortImageFilter;

template< typename ElementType, unsigned dimension >
class OutputPortImageFilter< Image< ElementType, dimension > >: public OutputPort
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > ImageType;

	OutputPortImageFilter(): _imageConnection( NULL ) {}

	//TODO - check const modifier
	ImageType&
	GetImage()const;

	void
	SetImageSize( 
			size_t 		minimums[ dimension ], 
			size_t 		maximums[ dimension ], 
			float32		elementExtents[ dimension ]
		    );

	void
	Plug( ConnectionInterface & connection );

	/*void
	PlugTyped( ImageConnection< ImageType > & connection );*/
	
	void
	UnPlug();

	bool
	IsPlugged()const
		{ return _imageConnection != NULL; }

	void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		);

protected:

	ImageConnection< ImageType >	*_imageConnection;
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
	~PortList() {}

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
	OutputPortList&
	operator=( const OutputPortList& );


	std::vector< OutputPort* >	_ports;

	MessageReceiverInterface	*_msgReceiver;	
};


//******************************************************************************

}/*namespace Imaging*/
}/*namespace M4D*/

#include "Imaging/Ports.tcc"

#endif /*_PORTS_H*/

