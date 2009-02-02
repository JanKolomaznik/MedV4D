/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConnectionInterface.tcc 
 * @{ 
 **/

#ifndef _CONNECTION_INTERFACE_H
#error File ConnectionInterface.tcc cannot be included directly!
#else

#include "Imaging/Ports.h"
/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

template< typename DatasetType >
void
ConnectionInterfaceTyped< DatasetType >
::ConnectProducer( OutputPort& outputPort )
{
	if( outputPort.IsConnectionCompatible( *this ) ) {
			outputPort.Plug( *this );
			this->_producer = &outputPort;
	} else {
		_THROW_ ConnectionInterface::EMismatchPortType();
	}
}

template< typename DatasetType >
void
ConnectionInterfaceTyped< DatasetType >
::ConnectConsumer( InputPort& inputPort )
{
	if( inputPort.IsConnectionCompatible( *this ) )
	{
		this->PushConsumer( inputPort );
	} else {
		_THROW_ ConnectionInterface::EMismatchPortType();
	}
}

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_CONNECTION_INTERFACE_H*/

/** @} */

