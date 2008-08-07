

#include <string>
#include <sstream>
#include <vector>

#include <iostream>

#include "Common.h"
#include "cellBE/netcommons.h"
#include "cellBE/clientSocket.h"

using namespace M4D::CellBE;
using namespace std;
using boost::asio::ip::tcp;

///////////////////////////////////////////////////////////////////////////////

ClientSocket::ClientSocket( const std::string &address,
  boost::asio::io_service &service)
  : m_address(address), BasicJob( service)
{
  Connect( service);
}

///////////////////////////////////////////////////////////////////////////////

void
ClientSocket::Connect( boost::asio::io_service &service)
{
  tcp::resolver resolver(service);

  stringstream port;
  port << SERVER_PORT;

  tcp::resolver::query query(
    m_address,
    port.str(),
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

///////////////////////////////////////////////////////////////////////////////

//void
//ClientSocket::SendJob( ClientJob *job)
//{
//  // prepare serialization of filters & settings
//  job->SerializeFiltersProperties();
//
//  PrimaryJobHeader::Serialize( &job->primHeader);
//
//  SecondaryJobHeader::Serialize( &job->secHeader);
//
//  // create vector of serialized information to pass to sigle send operation
//  vector<boost::asio::const_buffer> buffers;
//  buffers.push_back( 
//    boost::asio::buffer( (uint8*)&job->primHeader, sizeof(PrimaryJobHeader)) );
//  buffers.push_back( 
//    boost::asio::buffer( (uint8*)&job->secHeader, sizeof(SecondaryJobHeader)) );
//  buffers.push_back( 
//    boost::asio::buffer( 
//      &job->filterSettingsSerialized[0], job->filterSettingsSerialized.size() ));
//
//  // send the buffer vector
//  m_socket.async_write_some( 
//    buffers, 
//    boost::bind( &ClientSocket::EndSendJobHeader, 
//      this, boost::asio::placeholders::error, job)
//  );
//
//  //SendData( job);
//}

///////////////////////////////////////////////////////////////////////////////

//void
//ClientSocket::Resend( ClientJob *job)
//{
//  if( ! job->m_isPersistent)
//    throw ExceptionBase("Job is not persistent");
//}
//
/////////////////////////////////////////////////////////////////////////////////
//
//void
//ClientSocket::QuitJob( ClientJob *job)
//{
//  if( ! job->m_isPersistent)
//    throw ExceptionBase("Job is not persistent");
//}

///////////////////////////////////////////////////////////////////////////////

//void
//ClientSocket::OnJobResponseHeaderRead( 
//    const boost::system::error_code& err,
//    const size_t bytesRead, 
//    ClientJob *job)
//{
//  if( err)
//  {
//    job->state = ClientJob::Failed;
//    LOG( "OnJobResponseHeaderRead called with error code");
//    job->onComplete();
//  }
//  else
//  {
//    // now read response body
//    //std::istringstream stream( job.messageHeader.data);
//
//    //char *c = m_pok.c_array();
//
//    //uint16 messID;
//    //stream >> messID;
//
//    //uint16 messLen;
//    //stream >> messLen;
//
//    //job.response.resize( messLen);
//    //m_socket.async_read_some( boost::asio::buffer(job.response),
//    //  boost::bind( &ClientSocket::OnJobResponseBodyRead, this,
//    //    boost::asio::placeholders::error,
//    //    boost::asio::placeholders::bytes_transferred,
//    //    job) );
//  }
//}
//
/////////////////////////////////////////////////////////////////////////////////
//
//void
//ClientSocket::OnJobResponseBodyRead( 
//    const boost::system::error_code& err,
//    const size_t bytesRead, 
//    ClientJob *job)
//{
//  if( ! err)
//  {
//    job->state = ClientJob::Complete;
//  }
//  else
//  {
//    LOG( "OnJobResponseBodyRead called with error code");
//    job->state = ClientJob::Failed;
//  }
//
//  // call completition handler
//  if( job->onComplete) job->onComplete();
//}
//
/////////////////////////////////////////////////////////////////////////////////
//
//void
//ClientSocket::SendData( ClientJob *j)
//{
//  // iterate over data and create scatter buffer container
//  // containing pointers to data and pass it to network
//
//  //TODO
//  //m_socket.async_write_some(
//  //  boost::asio::buffer( j.),
//  //  boost::bind( &ClientSocket::EndSendData, this, 
//  //    boost::asio::placeholders::error,
//  //    j) );
//}
//
/////////////////////////////////////////////////////////////////////////////////
//
//void
//ClientSocket::EndSendJobHeader( 
//  const boost::system::error_code& e, ClientJob *j)
//{
//  if( e)
//  {
//    if( j->onError != NULL)
//      j->onError();
//  }
//  else
//    ReadResponseHeader( j);
//}
//
/////////////////////////////////////////////////////////////////////////////////
//
//void
//ClientSocket::EndSendData(
//  const boost::system::error_code& e, ClientJob *j)
//{
//  if( e)
//  {
//    if( j->onError != NULL)
//      j->onError();
//  }
//  else
//  {
//    // now we have to wait for response of da server
//
//    //TODO
//    //m_socket.async_read_some( boost::asio::buffer( m_pok),
//    //  boost::bind( &ClientSocket::OnJobResponseHeaderRead, this,
//    //    boost::asio::placeholders::error,
//    //    boost::asio::placeholders::bytes_transferred,
//    //    job) );
//  }
//}
//
///////////////////////////////////////////////////////////////////////////////
//
//void
//ClientSocket::ReadResponseHeader( ClientJob *job)
//{
//  m_socket.async_read_some( boost::asio::buffer((uint8*)&job->primHeader, sizeof(PrimaryJobHeader) ),
//    boost::bind( &ClientSocket::OnJobResponseBodyRead, this,
//      boost::asio::placeholders::error,
//      boost::asio::placeholders::bytes_transferred,
//      job) );
//}

///////////////////////////////////////////////////////////////////////////////
