/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file server.cpp 
 * @{ 
 **/

#include <sstream>

#include "common/Common.h"
#include "../server.h"
#include "../executionDoneCallback.h"
#include "../../netCommons.h"
#include "Imaging/DataSetFactory.h"

using asio::ip::tcp;
using namespace M4D::RemoteComputing;
using namespace M4D::Imaging;
using namespace M4D::IO;

///////////////////////////////////////////////////////////////////////////////

Server::Server(asio::io_service &io_service) :
	m_acceptor(io_service, tcp::endpoint(tcp::v4(), (uint16) SERVER_PORT) ),
			m_socket_(io_service), netAccessor_(m_socket_) {
	// start server accepting
	Accept();
}

///////////////////////////////////////////////////////////////////////////////

void Server::Accept(void) {	
	try {
		asio::error_code error;
		
		// and start accepting
		m_acceptor.accept(m_socket_, error);

		if(error)
		{
			LOG("Accept failed!, " << error.message());
			return;
		}

		LOG( "Accepted conn from:"
				<< m_socket_.remote_endpoint().address() );

		ReadCommand();

	} catch (asio::system_error &e) {
		LOG( "asio::system_error caught" << e.what() );
		return;
		
		if(m_socket_.is_open())
			m_socket_.close();
		if(e.code() == asio::error::eof )
		{
			OnClientDisconnected();
		}
	} catch( NetException &ne) {
		LOG( "NetException in Server::ReadCommand" << ne.what() );
	} catch( ExceptionBase &e) {
		LOG( "ExceptionBase in Server::EndPrimaryHeaderRead" << e.what() );
	}
}

///////////////////////////////////////////////////////////////////////////////

void Server::ReadCommand(void) {
	uint8 command;

	while (1) { // only diconnecting of client can break this loop
		netAccessor_.GetData( (void*) &command, sizeof(uint8));

		switch ( (eCommand) command) {
		case CREATE:
			D_PRINT("Recieved CREATE command");
			CreatePipeline();
			break;

		case EXEC:
			D_PRINT("Recieved EXEC command");
			ReadFilterProperties();
			break;

		case DATASET:
			D_PRINT("Recieved DATASET command");
			ReadDataSet();
			break;

		default:
			ASSERT(false)
			;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

void Server::OnClientDisconnected() {
	LOG( "Client disconnected, reseting pipe and accepting");
	m_pipeLine.Reset();
	Accept();
}

///////////////////////////////////////////////////////////////////////////////

void Server::CreatePipeline(void) {
	InStream stream(&netAccessor_);

	// perform deserialization    
	m_filter = RemoteFilterFactory::DeserializeFilter(stream, &m_props);
	// set starting by message
	m_filter->SetUpdateInvocationStyle(Imaging::AbstractPipeFilter::UIS_ON_CHANGE_BEGIN);
	m_pipeLine.AddFilter(m_filter);
}

///////////////////////////////////////////////////////////////////////////////

void Server::ReadDataSet(void) {
	InStream stream(&netAccessor_);

	// create the dataSet
	AbstractDataSet::Ptr inputDataSet = DataSetFactory::DeserializeDataset(stream);

	D_PRINT("Connecting recieved dataset into pipeline");
	// connect it to pipeline
	m_pipeLine.MakeInputConnection( *m_filter, 0, inputDataSet);

	// create and connect created output dataSet
	D_PRINT("Creating output connection")
	m_connWithOutputDataSet = &m_pipeLine.MakeOutputConnection( *m_filter, 0,
			true);

	// add message listener to be able catch execution done or failed messages
	m_connWithOutputDataSet->SetMessageHook(MessageReceiverInterface::Ptr(new ExecutionDoneCallback(this)) );
}

///////////////////////////////////////////////////////////////////////////////

void Server::ReadFilterProperties(void) {
	InStream stream(&netAccessor_);

	m_props->DeserializeProperties(stream);

	// and execute the pipeline. Actual exectution will wait to whole
	// dataSet unlock when whole dataSet is read (don't forget to do it!!)
	m_pipeLine.ExecuteFirstFilter();
}
///////////////////////////////////////////////////////////////////////////////

void Server::OnExecutionDone(void) {
	// send resulting dataSet back
	uint8 result = (eRemoteComputationResult) OK;
	OutStream stream(&netAccessor_);
	stream.Put<uint8>(result);

	DataSetFactory::SerializeDataset(stream, m_connWithOutputDataSet->GetDataset());
//	m_connWithOutputDataSet->GetDataset().SerializeProperties(stream);
//	m_connWithOutputDataSet->GetDataset().SerializeData(stream);
}

///////////////////////////////////////////////////////////////////////////////
void Server::OnExecutionFailed(void) {
	uint8 result = (eRemoteComputationResult) FAILED;
	OutStream stream(&netAccessor_);
	stream.Put<uint8>(result);
}

///////////////////////////////////////////////////////////////////////////////

/** @} */

