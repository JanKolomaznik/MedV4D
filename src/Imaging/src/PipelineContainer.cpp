#include "Imaging/PipelineContainer.h"
#include "Imaging/ImagePorts.h"

#include <algorithm>

#include "Functors.h"

namespace M4D
{
namespace Imaging
{

ConnectionInterface *
CreateConnectionObjectFromPorts( OutputPort& outPort, InputPort& inPort, bool ownsDataset )
{
	//TODO better exceptions
	ConnectionInterface *connection = NULL;

	try {
		//checking if we have image ports
		OutputPortAbstractImage & oPort = dynamic_cast< OutputPortAbstractImage &> ( outPort );
		InputPortAbstractImage & iPort = dynamic_cast< InputPortAbstractImage &> ( inPort ); 
		
		int typeID = oPort.ImageGetElementTypeID();
		unsigned dim = oPort.ImageGetDimension();

		if( dim == 0 || typeID == NTID_UNKNOWN || iPort.ImageGetDimension() != dim 
			|| iPort.ImageGetElementTypeID() != typeID ) 
		{
			throw EAutoConnectingFailed();
		}
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( typeID, 
				DIMENSION_TEMPLATE_SWITCH_MACRO( dim, connection = new ImageConnection< Image< TTYPE, DIM > >( ownsDataset ); ) );

	}	
	catch ( ... ) {
		throw EAutoConnectingFailed();
	}

	return connection;
}

ConnectionInterface *
CreateConnectionObjectFromInputPort( InputPort& inPort, bool ownsDataset )
{
	ConnectionInterface *connection = NULL;
	try {
		InputPortAbstractImage & iPort = dynamic_cast< InputPortAbstractImage &> ( inPort ); 
		
		int typeID = iPort.ImageGetElementTypeID();
		unsigned dim = iPort.ImageGetDimension();

		if( dim == 0 || typeID == NTID_UNKNOWN ) 
		{
			throw EAutoConnectingFailed();
		}
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( typeID, 
				DIMENSION_TEMPLATE_SWITCH_MACRO( dim, connection = new ImageConnection< Image< TTYPE, DIM > >( ownsDataset ); ) );

	}	
	catch ( ... ) {
		throw EAutoConnectingFailed();
	}
	return connection;
}

ConnectionInterface *
CreateConnectionObjectFromOutputPort( OutputPort& outPort, bool ownsDataset )
{
	ConnectionInterface *connection = NULL;
	try {
		OutputPortAbstractImage & oPort = dynamic_cast< OutputPortAbstractImage &> ( outPort );
		
		int typeID = oPort.ImageGetElementTypeID();
		unsigned dim = oPort.ImageGetDimension();

		if( dim == 0 || typeID == NTID_UNKNOWN ) 
		{
			throw EAutoConnectingFailed();
		}
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( typeID, 
				DIMENSION_TEMPLATE_SWITCH_MACRO( dim, connection = new ImageConnection< Image< TTYPE, DIM > >( ownsDataset ); ) );

	}	
	catch ( ... ) {
		throw EAutoConnectingFailed();
	}
	return connection;
}

//*********************************************************

PipelineContainer::PipelineContainer()
{

}

PipelineContainer::~PipelineContainer()
{
	std::for_each( _connections.begin(), _connections.end(), Functors::Deletor< ConnectionInterface *>() );

	std::for_each( _filters.begin(), _filters.end(), Functors::Deletor< AbstractPipeFilter *>() );
}

void
PipelineContainer::AddFilter( AbstractPipeFilter *filter )
{
	if( filter == NULL ) {
		throw ErrorHandling::ENULLPointer();
	}

	_filters.push_back( filter );
}

void
PipelineContainer::AddConnection( ConnectionInterface *connection )
{
	if( connection == NULL ) {
		throw ErrorHandling::ENULLPointer();
	}

	_connections.push_back( connection );
}

ConnectionInterface &
PipelineContainer::MakeConnection( M4D::Imaging::OutputPort& outPort, M4D::Imaging::InputPort& inPort )
{
		//if inPort occupied - error. Connection concept is designed only one to many.
	if( inPort.IsPlugged() ) {
		throw Port::EPortAlreadyConnected();
	}

	ConnectionInterface *connection = NULL;
	//if outPort is connected, we use already created Conncetion, otherwise we 
	//have to create new one.
	if( outPort.IsPlugged() ) {
		//TODO -check
		connection = outPort.GetConnection();
	} else {

		//TODO
		connection = CreateConnectionObjectFromPorts( outPort, inPort, true );
		//Newly created connection will be stored.
		_connections.push_back( connection );
	}

	connection->ConnectConsumer( inPort );
	connection->ConnectProducer( outPort );

	return *connection;
	
}

ConnectionInterface &
PipelineContainer::MakeConnection( M4D::Imaging::AbstractPipeFilter& producer, unsigned producerPortNumber, 
		M4D::Imaging::AbstractPipeFilter& consumer, unsigned consumerPortNumber )
{
	return MakeConnection( producer.OutputPort()[ producerPortNumber ], consumer.InputPort()[ consumerPortNumber ] );
}

ConnectionInterface &
PipelineContainer::MakeInputConnection( M4D::Imaging::InputPort& inPort, bool ownsDataset )
{
	if( inPort.IsPlugged() ) {
		throw Port::EPortAlreadyConnected();
	}

	ConnectionInterface *connection = NULL;

	connection = CreateConnectionObjectFromInputPort( inPort, ownsDataset );
	connection->ConnectConsumer( inPort );

	_connections.push_back( connection );
	return *connection;
}

ConnectionInterface &
PipelineContainer::MakeInputConnection(  M4D::Imaging::AbstractPipeFilter& consumer, unsigned consumerPortNumber, bool ownsDataset )
{
	return MakeInputConnection( consumer.InputPort()[ consumerPortNumber ], ownsDataset );
}

ConnectionInterface &
PipelineContainer::MakeInputConnection(  M4D::Imaging::AbstractPipeFilter& consumer, unsigned consumerPortNumber, AbstractDataSet::ADataSetPtr dataset )
{
	ConnectionInterface &conn =  MakeInputConnection( consumer.InputPort()[ consumerPortNumber ], false );
	conn.PutDataset( dataset );
	return conn;
}

ConnectionInterface &
PipelineContainer::MakeOutputConnection( M4D::Imaging::OutputPort& outPort, bool ownsDataset )
{
	if( outPort.IsPlugged() ) {
		throw Port::EPortAlreadyConnected();
	}

	ConnectionInterface *connection = NULL;

	connection = CreateConnectionObjectFromOutputPort( outPort, ownsDataset );
	connection->ConnectProducer( outPort );

	_connections.push_back( connection );
	return *connection;
}

ConnectionInterface &
PipelineContainer::MakeOutputConnection( M4D::Imaging::AbstractPipeFilter& producer, unsigned producerPortNumber, bool ownsDataset )
{
	return MakeOutputConnection( producer.OutputPort()[ producerPortNumber ], ownsDataset );
}

ConnectionInterface &
PipelineContainer::MakeOutputConnection( M4D::Imaging::AbstractPipeFilter& producer, unsigned producerPortNumber, AbstractDataSet::ADataSetPtr dataset )
{
	ConnectionInterface &conn = MakeOutputConnection( producer.OutputPort()[ producerPortNumber ], false );
	conn.PutDataset( dataset );
	return conn;
}

}/*namespace Imaging*/
}/*namespace M4D*/

