

#include <string>
#include <sstream>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/bind.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>

#include "Common.h"
#include <boost/serialization/vector.hpp>
#include "cellBE/netCommons.h"
#include "serverConn.h"

using namespace M4D::CellBE;
using namespace std;

///////////////////////////////////////////////////////////////////////

ServerConnection::ServerConnection( const std::string &address,
  boost::asio::io_service &service)
  : m_address(address), m_socket( service)
{
  Connect( service);
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::Connect( boost::asio::io_service &service)
{
  tcp::resolver resolver(service);

  char portNumBuf[6];

  try {
    _itoa_s<6>( M4D::CellBE::SERVER_PORT, portNumBuf, 10);
  } catch( ... ) {}

  tcp::resolver::query query(
    m_address,
    std::string( portNumBuf),
    tcp::resolver::query::numeric_service);

  tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
  tcp::resolver::iterator end;

  boost::system::error_code error = boost::asio::error::host_not_found;
  while (error && endpoint_iterator != end)
  {
    m_socket.close();
    m_socket.connect(*endpoint_iterator++, error);
  }
  if (error)
    throw M4D::ErrorHandling::ExceptionBase(
      "Not able to connect to Cell sever");
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::SendJob( ClientJob *job)
{
  // prepare serialization of filters & settings
  job->SerializeFiltersSetting();

  PrimaryJobHeader::Serialize( &job->primHeader);

  SecondaryJobHeader::Serialize( &job->secHeader);

  // create vector of serialized information to pass to sigle send operation
  vector<boost::asio::const_buffer> buffers;
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&job->primHeader, sizeof(PrimaryJobHeader)) );
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&job->secHeader, sizeof(SecondaryJobHeader)) );
  buffers.push_back( 
    boost::asio::buffer( 
      &job->filterSettingsSerialized[0], job->filterSettingsSerialized.size() ));

  // send the buffer vector
  m_socket.async_write_some( 
    buffers, 
    boost::bind( &ServerConnection::EndSendJobHeader, 
      this, boost::asio::placeholders::error, job)
  );

  //SendData( job);
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::Resend( ClientJob *job)
{
  if( ! job->m_isPersistent)
    throw ExceptionBase("Job is not persistent");
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::QuitJob( ClientJob *job)
{
  if( ! job->m_isPersistent)
    throw ExceptionBase("Job is not persistent");
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::OnJobResponseHeaderRead( 
    const boost::system::error_code& err,
    const size_t bytesRead, 
    ClientJob *job)
{
  if( err)
  {
    job->state = ClientJob::Failed;
    LOG( "OnJobResponseHeaderRead called with error code");
    job->onComplete();
  }
  else
  {
    // now read response body
    //std::istringstream stream( job.messageHeader.data);

    //char *c = m_pok.c_array();

    //uint16 messID;
    //stream >> messID;

    //uint16 messLen;
    //stream >> messLen;

    //job.response.resize( messLen);
    //m_socket.async_read_some( boost::asio::buffer(job.response),
    //  boost::bind( &ServerConnection::OnJobResponseBodyRead, this,
    //    boost::asio::placeholders::error,
    //    boost::asio::placeholders::bytes_transferred,
    //    job) );
  }
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::OnJobResponseBodyRead( 
    const boost::system::error_code& err,
    const size_t bytesRead, 
    ClientJob *job)
{
  if( ! err)
  {
    job->state = ClientJob::Complete;
  }
  else
  {
    LOG( "OnJobResponseBodyRead called with error code");
    job->state = ClientJob::Failed;
  }

  // call completition handler
  if( job->onComplete) job->onComplete();
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::SendData( ClientJob *j)
{
  // iterate over data and create scatter buffer container
  // containing pointers to data and pass it to network

  //TODO
  //m_socket.async_write_some(
  //  boost::asio::buffer( j.),
  //  boost::bind( &ServerConnection::EndSendData, this, 
  //    boost::asio::placeholders::error,
  //    j) );
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::EndSendJobHeader( 
  const boost::system::error_code& e, ClientJob *j)
{
  if( e)
  {
    if( j->onError != NULL)
      j->onError();
  }
  else
    ReadResponseHeader( j);
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::EndSendData(
  const boost::system::error_code& e, ClientJob *j)
{
  if( e)
  {
    if( j->onError != NULL)
      j->onError();
  }
  else
  {
    // now we have to wait for response of da server

    //TODO
    //m_socket.async_read_some( boost::asio::buffer( m_pok),
    //  boost::bind( &ServerConnection::OnJobResponseHeaderRead, this,
    //    boost::asio::placeholders::error,
    //    boost::asio::placeholders::bytes_transferred,
    //    job) );
  }
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::ReadResponseHeader( ClientJob *job)
{
  m_socket.async_read_some( boost::asio::buffer((uint8*)&job->primHeader, sizeof(PrimaryJobHeader) ),
    boost::bind( &ServerConnection::OnJobResponseBodyRead, this,
      boost::asio::placeholders::error,
      boost::asio::placeholders::bytes_transferred,
      job) );
}

///////////////////////////////////////////////////////////////////////
