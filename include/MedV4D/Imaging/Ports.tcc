/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Ports.tcc 
 * @{ 
 **/

#ifndef _PORTS_H
#error File Ports.tcc cannot be included directly!
#else

#include "MedV4D/Imaging/ConnectionInterface.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

template< typename DatasetType >
const DatasetType&
InputPortTyped< DatasetType >
::GetDatasetTyped()const
{
	if( this->_connection == NULL ) {
		_THROW_ Port::EDisconnected( *this );
	}
	return static_cast< IdealConnectionInterface *>(this->_connection)->GetDatasetReadOnlyTyped();
}

template< typename DatasetType >
typename DatasetType::ConstPtr
InputPortTyped< DatasetType >
::GetDatasetTypedPtr()const
{
	if( this->_connection == NULL ) {
		_THROW_ Port::EDisconnected( *this );
	}
	return static_cast< IdealConnectionInterface *>(this->_connection)->GetDatasetReadOnlyTypedPtr();
}

template< typename DatasetType >
void
InputPortTyped< DatasetType >
::Plug( ConnectionInterface & connection )
{
	if( this->IsPlugged() ) {
		_THROW_ Port::EPortAlreadyConnected();
	}

	if( IsConnectionCompatible( connection ) ) {
		this->_connection = &connection;
		this->PortPluggedMsg();
	} else {
		_THROW_ Port::EConnectionTypeMismatch();
	}
}

template< typename DatasetType >
ConnectionInterface*
InputPortTyped< DatasetType >
::CreateIdealConnectionObject( bool ownsDataset )
{
	return new ConnectionTyped< DatasetType >( ownsDataset );
}

template< typename DatasetType >
bool
InputPortTyped< DatasetType >
::IsConnectionCompatible( ConnectionInterface &conn )
{ 
	return dynamic_cast< IdealConnectionInterface * >( &conn ) != NULL; 
}
//******************************************************************************
template< typename DatasetType >
DatasetType&
OutputPortTyped< DatasetType >
::GetDatasetTyped()const
{
	if( this->_connection == NULL ) {
		_THROW_ Port::EDisconnected( *this );
	}
	return static_cast< IdealConnectionInterface *>(this->_connection)->GetDatasetTyped();
}

template< typename DatasetType >
typename DatasetType::Ptr
OutputPortTyped< DatasetType >
::GetDatasetTypedPtr()const
{
	if( this->_connection == NULL ) {
		_THROW_ Port::EDisconnected( *this );
	}
	return static_cast< IdealConnectionInterface *>(this->_connection)->GetDatasetTypedPtr();
}

template< typename DatasetType >
void
OutputPortTyped< DatasetType >
::Plug( ConnectionInterface & connection )
{
	if( this->IsPlugged() ) {
		_THROW_ Port::EPortAlreadyConnected();
	}

	if( IsConnectionCompatible( connection ) ) {
		this->_connection = &connection;
		this->PortPluggedMsg();
	} else {
		_THROW_ Port::EConnectionTypeMismatch();
	}
}

template< typename DatasetType >
ConnectionInterface*
OutputPortTyped< DatasetType >
::CreateIdealConnectionObject( bool ownsDataset )
{
	return new ConnectionTyped< DatasetType >( ownsDataset );
}

template< typename DatasetType >
bool
OutputPortTyped< DatasetType >
::IsConnectionCompatible( ConnectionInterface &conn )
{ 
	return dynamic_cast< IdealConnectionInterface * >( &conn ) != NULL; 
}
//******************************************************************************
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
//******************************************************************************

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

/** @} */

#endif /*_PORTS_H*/

/** @} */

