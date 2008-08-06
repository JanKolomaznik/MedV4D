#ifndef _PORTS_H
#error File Ports.tcc cannot be included directly!
#else

#include "Imaging/ConnectionInterface.h"

namespace M4D
{
namespace Imaging
{

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

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_PORTS_H*/
