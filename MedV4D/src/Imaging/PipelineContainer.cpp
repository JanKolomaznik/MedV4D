/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file PipelineContainer.cpp
 * @{
 **/

#include "MedV4D/Imaging/PipelineContainer.h"
#include "MedV4D/Imaging/Ports.h"

#include <algorithm>

#include "MedV4D/Common/Functors.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  @author Jan Kolomaznik
 */

namespace M4D
{
namespace Imaging {

ConnectionInterface *
CreateConnectionObjectFromPorts ( OutputPort& outPort, InputPort& inPort, bool ownsDataset )
{
        //TODO better exceptions
        ConnectionInterface *connection = NULL;

        if ( outPort.GetHierarchyDepth() > inPort.GetHierarchyDepth() ) {
                connection = outPort.CreateIdealConnectionObject ( ownsDataset );
                if ( !inPort.IsConnectionCompatible ( *connection ) ) {
                        delete connection;
                        _THROW_ EAutoConnectingFailed();
                }
        } else {
                connection = inPort.CreateIdealConnectionObject ( ownsDataset );
                if ( !outPort.IsConnectionCompatible ( *connection ) ) {
                        delete connection;
                        _THROW_ EAutoConnectingFailed();
                }
        }
        /*try {
        	//checking if we have image ports
        	OutputPortAImage & oPort = dynamic_cast< OutputPortAImage &> ( outPort );
        	InputPortAImage & iPort = dynamic_cast< InputPortAImage &> ( inPort );

        	int typeID = oPort.ImageGetElementTypeID();
        	unsigned dim = oPort.ImageGetDimension();

        	if( dim == 0 || typeID == NTID_UNKNOWN || iPort.ImageGetDimension() != dim
        		|| iPort.ImageGetElementTypeID() != typeID )
        	{
        		_THROW_ EAutoConnectingFailed();
        	}

        	TYPE_TEMPLATE_SWITCH_MACRO( typeID,
        			DIMENSION_TEMPLATE_SWITCH_MACRO( dim, connection = new ImageConnection< Image< TTYPE, DIM > >( ownsDataset ); ) );

        }
        catch ( ... ) {
        	_THROW_ EAutoConnectingFailed();
        }*/

        return connection;
}

ConnectionInterface *
CreateConnectionObjectFromInputPort ( InputPort& inPort, bool ownsDataset )
{
        ConnectionInterface *connection = NULL;
        connection = inPort.CreateIdealConnectionObject ( ownsDataset );
        /*try {
        	InputPortAImage & iPort = dynamic_cast< InputPortAImage &> ( inPort );

        	int typeID = iPort.ImageGetElementTypeID();
        	unsigned dim = iPort.ImageGetDimension();

        	if( dim == 0 || typeID == NTID_UNKNOWN )
        	{
        		connection = new AImageConnection();
        	} else {
        		TYPE_TEMPLATE_SWITCH_MACRO( typeID,
        			DIMENSION_TEMPLATE_SWITCH_MACRO( dim, connection = new ImageConnection< Image< TTYPE, DIM > >( ownsDataset ); ) );
        	}

        }
        catch ( ... ) {
        	_THROW_ EAutoConnectingFailed();
        }*/
        return connection;
}

ConnectionInterface *
CreateConnectionObjectFromOutputPort ( OutputPort& outPort, bool ownsDataset )
{
        ConnectionInterface *connection = NULL;
        connection = outPort.CreateIdealConnectionObject ( ownsDataset );
        /*try {
        	OutputPortAImage & oPort = dynamic_cast< OutputPortAImage &> ( outPort );

        	int typeID = oPort.ImageGetElementTypeID();
        	unsigned dim = oPort.ImageGetDimension();

        	if( dim == 0 || typeID == NTID_UNKNOWN )
        	{
        		_THROW_ EAutoConnectingFailed();
        	}

        	TYPE_TEMPLATE_SWITCH_MACRO( typeID,
        			DIMENSION_TEMPLATE_SWITCH_MACRO( dim, connection = new ImageConnection< Image< TTYPE, DIM > >( ownsDataset ); ) );

        }
        catch ( ... ) {
        	_THROW_ EAutoConnectingFailed();
        }*/
        return connection;
}

//*********************************************************

PipelineContainer::PipelineContainer()
{

}

PipelineContainer::~PipelineContainer()
{
        std::for_each ( _filters.begin(), _filters.end(), Functors::Deletor< APipeFilter *>() );

        std::for_each ( _connections.begin(), _connections.end(), Functors::Deletor< ConnectionInterface *>() );
}

void
PipelineContainer::Reset()
{
        std::for_each ( _filters.begin(), _filters.end(), Functors::Deletor< APipeFilter *>() );
        _filters.clear();

        std::for_each ( _connections.begin(), _connections.end(), Functors::Deletor< ConnectionInterface *>() );
        _connections.clear();

}

void
PipelineContainer::AddFilter ( APipeFilter *filter )
{
        if ( filter == NULL ) {
                _THROW_ ErrorHandling::ENULLPointer();
        }

        _filters.push_back ( filter );
}

void
PipelineContainer::AddConnection ( ConnectionInterface *connection )
{
        if ( connection == NULL ) {
                _THROW_ ErrorHandling::ENULLPointer();
        }

        _connections.push_back ( connection );
}

void
PipelineContainer::StopFilters()
{
        for ( unsigned i = 0; i < _filters.size(); ++i ) {
                _filters[i]->StopExecution();
        }
}

ConnectionInterface &
PipelineContainer::MakeConnection ( M4D::Imaging::OutputPort& outPort, M4D::Imaging::InputPort& inPort )
{
        //if inPort occupied - error. Connection concept is designed only one to many.
        if ( inPort.IsPlugged() ) {
                _THROW_ Port::EPortAlreadyConnected();
        }

        ConnectionInterface *connection = NULL;
        //if outPort is connected, we use already created Conncetion, otherwise we
        //have to create new one.
        if ( outPort.IsPlugged() ) {
                //TODO -check
                connection = outPort.GetConnection();
        } else {

                //TODO
                connection = CreateConnectionObjectFromPorts ( outPort, inPort, true );
                //Newly created connection will be stored.
                _connections.push_back ( connection );

                connection->ConnectProducer ( outPort );
        }

        connection->ConnectConsumer ( inPort );

        return *connection;

}

ConnectionInterface &
PipelineContainer::MakeConnection ( M4D::Imaging::APipeFilter& producer, unsigned producerPortNumber,
                                    M4D::Imaging::APipeFilter& consumer, unsigned consumerPortNumber )
{
        return MakeConnection ( producer.OutputPort() [ producerPortNumber ], consumer.InputPort() [ consumerPortNumber ] );
}

ConnectionInterface &
PipelineContainer::MakeInputConnection ( M4D::Imaging::InputPort& inPort, bool ownsDataset )
{
        if ( inPort.IsPlugged() ) {
                _THROW_ Port::EPortAlreadyConnected();
        }

        ConnectionInterface *connection = NULL;

        connection = CreateConnectionObjectFromInputPort ( inPort, ownsDataset );
        connection->ConnectConsumer ( inPort );

        _connections.push_back ( connection );
        return *connection;
}

ConnectionInterface &
PipelineContainer::MakeInputConnection ( M4D::Imaging::APipeFilter& consumer, unsigned consumerPortNumber, bool ownsDataset )
{
        return MakeInputConnection ( consumer.InputPort() [ consumerPortNumber ], ownsDataset );
}

ConnectionInterface &
PipelineContainer::MakeInputConnection ( M4D::Imaging::APipeFilter& consumer, unsigned consumerPortNumber, ADataset::Ptr dataset )
{
        ConnectionInterface &conn =  MakeInputConnection ( consumer.InputPort() [ consumerPortNumber ], false );
        conn.PutDataset ( dataset );
        return conn;
}

ConnectionInterface &
PipelineContainer::MakeOutputConnection ( M4D::Imaging::OutputPort& outPort, bool ownsDataset )
{
        if ( outPort.IsPlugged() ) {
                _THROW_ Port::EPortAlreadyConnected();
        }

        ConnectionInterface *connection = NULL;

        connection = CreateConnectionObjectFromOutputPort ( outPort, ownsDataset );
        connection->ConnectProducer ( outPort );

        _connections.push_back ( connection );
        return *connection;
}

ConnectionInterface &
PipelineContainer::MakeOutputConnection ( M4D::Imaging::APipeFilter& producer, unsigned producerPortNumber, bool ownsDataset )
{
        return MakeOutputConnection ( producer.OutputPort() [ producerPortNumber ], ownsDataset );
}

ConnectionInterface &
PipelineContainer::MakeOutputConnection ( M4D::Imaging::APipeFilter& producer, unsigned producerPortNumber, ADataset::Ptr dataset )
{
        ConnectionInterface &conn = MakeOutputConnection ( producer.OutputPort() [ producerPortNumber ], false );
        conn.PutDataset ( dataset );
        return conn;
}

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

