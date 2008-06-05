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
class OutConnectionInterface;
class InConnectionInterface;

template< typename ImageTemplate >
class ImageConnection;

//**************************************

class Port: public MessageOperatorInterface
{
public:
	/** 
	 * Type of ports identification number. 
	 **/
	typedef	uint64 PortID;

	/**
	 * Exception which is thrown in situations, when port isn't
	 * connected and method need port to be connected for succesful
	 * execution.
	 **/
	class EDisconnected;

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
	
	/**
	 * Method to unplug port from connection object - if already 
	 * disconnected do nothing.
	 **/
	virtual void
	UnPlug() = 0;

	
protected:
	/**
	 * Method for unique port ID generation - thread safe.
	 **/
	static PortID
	GenerateUniqueID();

	PortID	_id;
private:

};

/**
 * Exception which is thrown in situations, when port isn't
 * connected and method need port to be connected for succesful
 * execution.
 **/
class Port::EDisconnected
{
public:
	EDisconnected( Port::PortID port ) : _port( port ) {}
	//TODO
protected:
	/**
	 * ID of port which caused this exception.
	 **/
	Port::PortID _port;
};



class InputPort: public Port
{
public:
	virtual
	~InputPort() {}

	virtual void
	Plug( OutConnectionInterface & connection ) = 0;
protected:
	virtual void
	SetPlug( OutConnectionInterface & connection ) = 0;
private:

};

class OutputPort: public Port
{
public:
	virtual
	~OutputPort() {}

	virtual void
	Plug( InConnectionInterface & connection ) = 0;
protected:
	virtual void
	SetPlug( InConnectionInterface & connection ) = 0;

private:

};

//******************************************************************************
template< typename ImageType >
class InputPortImageFilter;

template< typename ElementType, unsigned dimension >
class InputPortImageFilter< Image< ElementType, dimension > >: public InputPort
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > ImageType;

	InputPortImageFilter(): _imageConnection( NULL ) {}

	const ImageType&
	GetImage()const;
	
	void
	Plug( OutConnectionInterface & connection );

	void
	Plug( ImageConnection< ImageType > & connection );

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

	void
	ReceiveMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle, 
		FlowDirection				direction
		);

protected:
	void
	SetPlug( OutConnectionInterface & connection );
	void
	SetPlug( ImageConnection< ImageType > & connection );
	
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
	Plug( InConnectionInterface & connection );

	void
	Plug( ImageConnection< ImageType > & connection );
	
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

	void
	ReceiveMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle, 
		FlowDirection				direction
		);

protected:
	void
	SetPlug( InConnectionInterface & connection );
	void
	SetPlug( ImageConnection< ImageType > & connection );	

	ImageConnection< ImageType >	*_imageConnection;
};

//******************************************************************************

class PortList
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
	InputPortList() {}

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
private:
	//Not implemented
	InputPortList( const InputPortList& );
	InputPortList&
	operator=( const InputPortList& );


	std::vector< InputPort* >	_ports;
};

class OutputPortList: public PortList
{
public:
	OutputPortList() {}

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
private:
	//Not implemented
	OutputPortList( const OutputPortList& );
	OutputPortList&
	operator=( const OutputPortList& );


	std::vector< OutputPort* >	_ports;
};


//******************************************************************************

}/*namespace Imaging*/
}/*namespace M4D*/

#include "Imaging/Ports.tcc"

#endif /*_PORTS_H*/

