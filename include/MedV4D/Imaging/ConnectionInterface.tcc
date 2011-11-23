/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConnectionInterface.tcc 
 * @{ 
 **/

#ifndef _CONNECTION_INTERFACE_H
#error File ConnectionInterface.tcc cannot be included directly!
#else

#include "MedV4D/Imaging/Ports.h"
/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{
template< typename DatasetType, bool doConstruct >
struct DatasetConstructionHelper;

template< typename DatasetType >
struct DatasetConstructionHelper< DatasetType, true >
{
	static typename DatasetType::Ptr
	ConstructDatasetIfPossible()
	{ return typename DatasetType::Ptr( new DatasetType() ); }
};

template< typename DatasetType >
struct DatasetConstructionHelper< DatasetType, false >
{
	static typename DatasetType::Ptr
	ConstructDatasetIfPossible()
	{ return typename DatasetType::Ptr(); }
};

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

template< typename DatasetType >
ConnectionTyped< DatasetType >
::ConnectionTyped( bool ownsDataset )
{
	if( ownsDataset ) {
		_dataset = DatasetConstructionHelper< DatasetType, DatasetType::IsConstructable >::ConstructDatasetIfPossible();
	}
}

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_CONNECTION_INTERFACE_H*/

/** @} */

