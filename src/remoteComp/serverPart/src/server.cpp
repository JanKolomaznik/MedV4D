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

//#define ONLY_1_ACCEPT 1

///////////////////////////////////////////////////////////////////////////////

Server::Server(asio::io_service &io_service) 
	: m_acceptor(io_service, tcp::endpoint(tcp::v4(), (uint16) SERVER_PORT) )
	, m_socket_(io_service), netAccessor_(m_socket_) 
{
}

///////////////////////////////////////////////////////////////////////////////

void
Server::AcceptLoop()
{
	while(1)
	{
		Accept();
#ifdef ONLY_1_ACCEPT
		break;
#endif
	}
}

///////////////////////////////////////////////////////////////////////////////

void Server::Accept(void) 
{	
	try {
		
		// and start accepting
		m_acceptor.accept(m_socket_, _error);

		if(_error)
		{
			LOG("Accept failed!, " << _error.message());
			return;
		}

		LOG( "Accepted conn from:"
				<< m_socket_.remote_endpoint().address() );

		ReadCommand();

	} catch (asio::system_error &e) {
		LOG( "asio::system_error: " << e.what() );
		
		if(e.code() == asio::error::eof )
		{
			OnClientDisconnected();
		}
	} catch( NetException &ne) {
		LOG( "NetException in Server::ReadCommand" << ne.what() );
	} catch( ExceptionBase &e) {
		LOG( "ExceptionBase in Server::EndPrimaryHeaderRead" << e.what() );
	} catch( ... ) {
		LOG( "UNKNOWN exception in Server::EndPrimaryHeaderRead");
	}
	
	if(m_socket_.is_open())
		m_socket_.close();
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
}

///////////////////////////////////////////////////////////////////////////////

void Server::CreatePipeline(void) {
	InStream stream(&netAccessor_);

	D_PRINT("Deserializing filter definition ...");	
	// perform deserialization    
	m_filter = RemoteFilterFactory::DeserializeFilterClassID(stream, &m_props);
	D_PRINT("done");
	// set starting by message
	m_filter->SetUpdateInvocationStyle(Imaging::AbstractPipeFilter::UIS_ON_CHANGE_BEGIN);
	m_pipeLine.AddFilter(m_filter);
	
	// create and connect created output dataSet
	m_connWithOutputDataSet = &m_pipeLine.MakeOutputConnection( *m_filter, 0,
			true);

	// add message listener to be able catch execution done or failed messages
	m_connWithOutputDataSet->SetMessageHook(MessageReceiverInterface::Ptr(new ExecutionDoneCallback(this)) );
	
	// create input connection
	_inputConnection = &m_pipeLine.MakeInputConnection( *m_filter, 0, false);
}

///////////////////////////////////////////////////////////////////////////////

void Server::ReadDataSet(void) 
{
	InStream stream(&netAccessor_);

	// create the dataSet
	D_PRINT("Deserializing data set");
	AbstractDataSet::Ptr inputDataSet = DataSetFactory::DeserializeDataset(stream);
	D_PRINT("done");
	
	_inputConnection->PutDataset(inputDataSet);
}

///////////////////////////////////////////////////////////////////////////////

void Server::ReadFilterProperties(void) {
	InStream stream(&netAccessor_);

	D_PRINT("Deserializing filter properties");
	RemoteFilterFactory::DeSerializeFilterProperties(stream, *m_props);
	D_PRINT("done");

	// and execute the pipeline. Actual exectution will wait to whole
	// dataSet unlock when whole dataSet is read (don't forget to do it!!)
	m_pipeLine.ExecuteFirstFilter();
}
///////////////////////////////////////////////////////////////////////////////

void Server::OnExecutionDone(void) {
	// save DS to file
	{
		M4D::IO::FOutStream outStr("out.mv4d");		
		DataSetFactory::SerializeDataset(
				outStr, m_connWithOutputDataSet->GetDataset());
	}
	// send resulting dataSet back
	uint8 result = (eRemoteComputationResult) OK;
	OutStream stream(&netAccessor_);
	D_PRINT("Sending result:" << OK);
	stream.Put<uint8>(result);

	D_PRINT("Serializing OUT data set");
	DataSetFactory::SerializeDataset(stream, m_connWithOutputDataSet->GetDataset());
	D_PRINT("done");
}

///////////////////////////////////////////////////////////////////////////////
void Server::OnExecutionFailed(void) {
	uint8 result = (eRemoteComputationResult) FAILED;
	OutStream stream(&netAccessor_);
	D_PRINT("Sending result:" << FAILED);
	stream.Put<uint8>(result);
}

///////////////////////////////////////////////////////////////////////////////

/** @} */

