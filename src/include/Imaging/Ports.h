#ifndef _PORTS_H
#define _PORTS_H

#include <boost/shared_ptr.hpp>
#include "Imaging/AbstractDataSet.h"
#include "Imaging/Image.h"
#include <vector>

namespace M4D
{
namespace Imaging
{

//Forward declarations *****************
class OutConnectionInterface;
class InConnectionInterface;

template<  typename ImageTemplate >
class InImageConnection;

template<  typename ImageTemplate >
class OutImageConnection;
//**************************************

class PipelineMessage
{


};

class Port
{
public:
	typedef	uint64 PortID;

	virtual
	~Port() { UnPlug(); }

	virtual	bool
	IsPlugged()const = 0;
		
	virtual void
	UnPlug() = 0;

protected:
	static PortID
	GenerateUniqueID();
private:

};

class InputPort
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

class OutputPort
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
class InputPortImageFilter< Image< ElementType, dimension > >
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > ImageType;

	const ImageType&
	GetImage()const;
	
	void
	UnPlug();

	void
	Plug( OutConnectionInterface & connection );

	void
	Plug( OutImageConnection< ImageType > & connection );
	
protected:
	void
	SetPlug( OutConnectionInterface & connection );
	void
	SetPlug( OutImageConnection< ImageType > & connection );
	
};

template< typename ImageType >
class OutputPortImageFilter;

template< typename ElementType, unsigned dimension >
class OutputPortImageFilter< Image< ElementType, dimension > >
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > ImageType;

	//TODO - check const modifier
	ImageType&
	GetImage()const;

	void
	Plug( InConnectionInterface & connection );

	void
	Plug( InImageConnection< ImageType > & connection );
	
protected:
	void
	SetPlug( InConnectionInterface & connection );
	void
	SetPlug( InImageConnection< ImageType > & connection );	
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

