

#include <string>
#include <sstream>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/bind.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>

#include "Common.h"
#include <boost/serialization/vector.hpp>
#include "cellBE/commDefs.h"
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
ServerConnection::SendJob( Job & job)
{
  std::ostringstream s;
  boost::archive::text_oarchive arch( s);

  //s << job.m_filterID;
  job.m_f1 = 3.12345f;
  job.m_str = "predel";

  arch << (const Job &)job;

  job.sendedMessage = s.str();

  m_socket.async_write_some(
    boost::asio::buffer( job.sendedMessage),
    boost::bind( &ServerConnection::OnJobWritten, this, 
      boost::asio::placeholders::error,
      job) );
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::OnJobWritten( 
  const boost::system::error_code& err,
  Job &job )
{
  if( ! err)
  {
    // now we have to wait for response of da server
    m_socket.async_read_some( boost::asio::buffer( m_pok),
      boost::bind( &ServerConnection::OnJobResponseHeaderRead, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::bytes_transferred,
        job) );
  }
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::OnJobResponseHeaderRead( 
    const boost::system::error_code& err,
    const size_t bytesRead, 
    Job &job)
{
  if( err)
  {
    job.state = Job::Failed;
    LOG( "OnJobResponseHeaderRead called with error code");
    job.onComplete();
  }
  else
  {
    // now read response body
    std::istringstream stream( job.messageHeader.data);

    char *c = m_pok.c_array();

    uint16 messID;
    stream >> messID;

    uint16 messLen;
    stream >> messLen;

    job.response.resize( messLen);
    m_socket.async_read_some( boost::asio::buffer(job.response),
      boost::bind( &ServerConnection::OnJobResponseBodyRead, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::bytes_transferred,
        job) );
  }
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::OnJobResponseBodyRead( 
    const boost::system::error_code& err,
    const size_t bytesRead, 
    Job &job)
{
  if( ! err)
  {
    job.state = Job::Complete;
  }
  else
  {
    LOG( "OnJobResponseBodyRead called with error code");
    job.state = Job::Failed;
  }

  // call completition handler
  job.onComplete();
}
