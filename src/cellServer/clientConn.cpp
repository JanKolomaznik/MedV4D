

#include <string>
#include <sstream>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/bind.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>

#include "Common.h"
#include "cellBE/commDefs.hpp"
#include "ExceptionBase.h"
#include "clientConn.h"

using namespace M4D::CellBE;
using namespace std;

///////////////////////////////////////////////////////////////////////

ClientConnection::ClientConnection(boost::asio::io_service& io_service)
  : m_socket(io_service)
{
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::SendJob( Job &job)
{
  std::ostringstream s;
  boost::archive::text_oarchive archive( s);

  archive << job;

  /*m_socket->async_write_some(
    boost::asio::buffer(str),
    boost::bind( &ServerConnection::OnJobWritten, this, 
      boost::asio::placeholders::error,
      boost::asio::placeholders::bytes_transferred,
      job) );*/
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::OnJobWritten( 
  const boost::system::error_code& e,
  const uint16 bytesWritten,
  const Job &j )
{
  // now we have to wait for response of da server
  /*m_socket->async_read_some( boost::asio::buffer(j.respHeader.data, 2),
    boost::bind( &ServerConnection::OnJobResponseHeaderRead, this,
      boost::asio::placeholders::error,
      boost::asio::placeholders::bytes_transferred,
      j) );*/
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::OnJobResponseHeaderRead( 
    const boost::system::error_code& e,
    const uint16 bytesRead, 
    Job &j)
{
  if( e)
  {
    j.state = Job::Failed;
    LOG( "OnJobResponseHeaderRead called with error code");
    j.onComplete();
  }
  else
  {
    // now read response body
    /*j.response.resize( 2);
    m_socket->async_read_some( boost::asio::buffer(j.response),
      boost::bind( &ServerConnection::OnJobResponseBodyRead, this,
        boost::asio::placeholders::error,
        boost::asio::placeholders::bytes_transferred,
        j) );*/
  }
}

///////////////////////////////////////////////////////////////////////

void
ServerConnection::OnJobResponseBodyRead( 
    const boost::system::error_code& e,
    const uint16 bytesRead, 
    Job &j)
{
  if( ! e)
  {
    j.state = Job::Complete;
  }
  else
  {
    LOG( "OnJobResponseBodyRead called with error code");
    j.state = Job::Failed;
  }

  // call completition handler
  j.onComplete();
}
