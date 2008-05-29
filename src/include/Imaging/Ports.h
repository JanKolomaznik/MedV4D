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

class PipelineMessage
{


};

class Port
{
public:
	typedef	uint64 PortID;

	virtual
	~Port() {}
		
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

protected:

private:

};

class OutputPort
{
public:
	virtual
	~OutputPort() {}

protected:

private:

};

//******************************************************************************
template< typename ImageType >
class InputPortImageFilter
{
public:
	const ImageType&
	GetImage()const;
};

template< typename ImageType >
class OutputPortImageFilter
{
public:
	//TODO - check const modifier
	ImageType&
	GetImage()const;

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

template< typename PortType >
PortType&
InputPortList::GetPortTyped( size_t idx )const
{
	return static_cast< PortType& >( GetPort( idx ) );
}

template< typename PortType >
PortType*
InputPortList::GetPortTypedSafe( size_t idx )const
{
	return dynamic_cast< PortType* >( &(GetPort( idx )) );
}

template< typename PortType >
PortType&
OutputPortList::GetPortTyped( size_t idx )const
{
	return static_cast< PortType& >( GetPort( idx ) );
}

template< typename PortType >
PortType*
OutputPortList::GetPortTypedSafe( size_t idx )const
{
	return dynamic_cast< PortType* >( &(GetPort( idx )) );
}

//******************************************************************************

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_PORTS_H*/

