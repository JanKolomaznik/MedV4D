/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file server.cpp 
 * @{ 
 **/

#include <sstream>

#include "Common.h"
#include "../server.h"
#include "../executionDoneCallback.h"
#include "../../netCommons.h"
#include "Imaging/DataSetFactory.h"

using boost::asio::ip::tcp;
using namespace M4D::RemoteComputing;
using namespace M4D::Imaging;

///////////////////////////////////////////////////////////////////////////////

Server::Server(boost::asio::io_service &io_service)
  : m_acceptor(io_service, tcp::endpoint(tcp::v4(), (uint16) SERVER_PORT) )
  , m_socket_(io_service)
  , netAccessor_(m_socket_)
{
  // start server accepting
  Accept();
}

///////////////////////////////////////////////////////////////////////////////

void
Server::Accept( void)
{
  // and start accepting
  m_acceptor.async_accept(
    m_socket_,
    boost::bind(&Server::EndAccepted, this, boost::asio::placeholders::error) 
    );
}

///////////////////////////////////////////////////////////////////////////////

void
Server::EndAccepted( const boost::system::error_code& error)
{
  try {

    HandleErrors( error);
    
    LOG( "Accepted conn from:" 
          << m_socket_.remote_endpoint().address() );

    ReadCommand(); 

  } catch ( NetException &ne) {
    LOG("NetException in EndAccepted" << ne.what() );
  }

  // accept again
  //Accept();
}

///////////////////////////////////////////////////////////////////////////////

void
Server::ReadCommand(void)
{
  try {
    uint8 command;
    m_socket_.read_some( boost::asio::buffer( &command, sizeof(uint8)) );
    
    switch( (eCommand) command)
    {
    case CREATE:
    	CreatePipeline();
      break;

    case EXEC:
    	ReadFilterProperties();
      break;
      
    case DATASET:
    	ReadDataSet();
      break;
    
    default:
      ASSERT(false);
    }

  } catch( NetException &ne) {
    LOG( "NetException in Server::EndPrimaryHeaderRead" << ne.what() );
  } catch( ExceptionBase &e) {
    LOG( "ExceptionBase in Server::EndPrimaryHeaderRead" << e.what() );
  }

}

///////////////////////////////////////////////////////////////////////////////

void
Server::CreatePipeline(void)
{	
	Imaging::InStream stream(&netAccessor_);
	
    // perform deserialization    
	m_pipelineBegin = RemoteFilterFactory::DeserializeFilter(stream);
    m_pipeLine.AddFilter( m_pipelineBegin);
    
  // set starting by message
    m_pipelineBegin->SetUpdateInvocationStyle( 
        Imaging::AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
  
  // currently only 1 filter
  m_pipelineEnd = m_pipelineBegin;
}

///////////////////////////////////////////////////////////////////////////////

void
Server::CleenupPipeline(void)
{
	//m_pipeLine.Clean();
	m_pipelineEnd = m_pipelineBegin = NULL; 
}

///////////////////////////////////////////////////////////////////////////////

void
Server::ReadDataSet(void)
{
	Imaging::InStream stream(&netAccessor_);
	
    // create the dataSet
    AbstractDataSet::Ptr inputDataSet = DataSetFactory::CreateDataSet(stream);

    // connect it to pipeline
    m_pipeLine.MakeInputConnection( *m_pipelineBegin, 0, inputDataSet);

    // create and connect created output dataSet
    ConnectionInterface &conn = 
      m_pipeLine.MakeOutputConnection( *m_pipelineEnd, 0, true);
    
    // add message listener to be able catch execution done or failed messages
    conn.SetMessageHook( 
      MessageReceiverInterface::Ptr( new ExecutionDoneCallback(this) ) );    
}

///////////////////////////////////////////////////////////////////////////////

void
Server::ReadFilterProperties(void)
{
	Imaging::InStream stream(&netAccessor_);
	
	// and execute the pipeline. Actual exectution will wait to whole
    // dataSet unlock when whole dataSet is read (don't forget to do it!!)
    m_pipelineBegin->Execute();
}
///////////////////////////////////////////////////////////////////////////////

void
Server::OnExecutionDone( void)
{
	
}
    
///////////////////////////////////////////////////////////////////////////////
void
Server::OnExecutionFailed( void)
{
	
}
    
///////////////////////////////////////////////////////////////////////////////

void
Server::HandleErrors(const boost::system::error_code& error)
{
	
}

///////////////////////////////////////////////////////////////////////////////

/** @} */

