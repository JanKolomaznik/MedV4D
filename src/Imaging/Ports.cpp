#include <algorithm>

#include "Thread.h"
#include "Functors.h"
#include "Imaging/Ports.h"

namespace M4D
{
namespace Imaging
{

/* Port::PortID */ uint64
Port::GenerateUniqueID()
{
	static /* PortID */ uint64 lastID = 0;
	static Multithreading::Mutex genMutex;

	{	//We must synchronize to avoid multiple generation of one ID.
		Multithreading::ScopedLock lock( genMutex );
		return ++lastID;
	}
}
//******************************************************************************
	

InputPortList::~InputPortList()
{
	std::for_each( 
		_ports.begin(), 
		_ports.end(), 
		Functors::Deletor< InputPort* >() 
		);
}

size_t
InputPortList::AddPort( InputPort* port )
{
	if( port == NULL ) {
		//TODO - throw exception
		return static_cast< size_t >( -1 );
	}
	_ports.push_back( port );
	return _size++;
}

InputPort &
InputPortList::GetPort( size_t idx )const
{
	//TODO - check bounds - throw exception
	return *_ports[ idx ];
}


OutputPortList::~OutputPortList()
{
	std::for_each( 
		_ports.begin(), 
		_ports.end(), 
		Functors::Deletor< OutputPort* >() 
		);
}

size_t
OutputPortList::AddPort( OutputPort* port )
{
	if( port == NULL ) {
		//TODO - throw exception
		return static_cast< size_t >( -1 );
	}
	_ports.push_back( port );
	return _size++;
}

OutputPort &
OutputPortList::GetPort( size_t idx )const
{
	//TODO - check bounds - throw exception
	return *_ports[ idx ];
}


}/*namespace Imaging*/
}/*namespace M4D*/
