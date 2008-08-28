/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file server.cpp 
 * @{ 
 **/


#include <sstream>

#include "Common.h"
#include "server.h"

#include "cellBE/netCommons.h"
#include "cellBE/netstream.h"

using namespace M4D::CellBE;
using boost::asio::ip::tcp;

Pool<PrimaryJobHeader, 32> Server::m_headerPool;

///////////////////////////////////////////////////////////////////////////////

Server::Server(boost::asio::io_service &io_service)
  : m_acceptor(io_service, tcp::endpoint(tcp::v4(), (uint16) SERVER_PORT) )
{
  // prepare ping message content
#define PING_MESSAGE_CONTENT "Hi.This is cell server."
  {
    std::stringstream s;
    s << PING_MESSAGE_CONTENT;
    s << "(IP: " << m_acceptor.local_endpoint().address() << ", port:";
    s << m_acceptor.local_endpoint().port() << ")";
    std::string str = s.str();

    m_pingStream << (uint16) str.size();

    // copy ping message string content to m_pingStream
    for( std::string::iterator it=str.begin(); it != str.end(); it++)
      m_pingStream << (uint8) *it;
  }

  // start server accepting
  Accept();
}

///////////////////////////////////////////////////////////////////////////////

void
Server::Accept( void)
{
  // create new instance of OneClientConnection
  //OneClientConnection *newConnection = 
  //  new OneClientConnection(m_acceptor.io_service());
  tcp::socket *sock = new tcp::socket( m_acceptor.io_service());

  // and start accepting
  m_acceptor.async_accept(
    *sock,
    boost::bind(&Server::EndAccepted, this, sock,
        boost::asio::placeholders::error) );
}

///////////////////////////////////////////////////////////////////////////////

void
Server::EndAccepted( tcp::socket *clientSock,
      const boost::system::error_code& error)
{
  try {

    BasicSocket::HandleErrors( error);

    PrimaryJobHeader *freeHeader = m_headerPool.GetFreeItem();

    clientSock->async_read_some(
      boost::asio::buffer( (uint8*)freeHeader, sizeof(PrimaryJobHeader) ),
      boost::bind( &Server::EndPrimaryHeaderRead, this, clientSock, freeHeader,
        boost::asio::placeholders::error)
      );

    LOG( "Accepted conn from:" 
      << clientSock->remote_endpoint().address() );

  } catch ( NetException &ne) {
    LOG("NetException in EndAccepted" << ne.what() );
  }

  // accept again
  Accept();
}

///////////////////////////////////////////////////////////////////////////////

void
Server::EndPrimaryHeaderRead( tcp::socket *clientSock, PrimaryJobHeader *header,
        const boost::system::error_code& error)
{
  try {
    BasicSocket::HandleErrors( error);

    // parse primary job header
    PrimaryJobHeader::Deserialize( header);

    ServerJob *existing;

    // create, remove & ping requests are dispatched here. Others are issued to
    // appropriate job
    switch( (BasicJob::Action) header->action)
    {
    case BasicJob::CREATE:
      LOG( "CREATE reqest arrived");

      existing = new ServerJob( clientSock, &m_jobManager);
      existing->primHeader = *header;

      m_jobManager.AddJob( existing);
      existing->ReadPipelineDefinition();
      break;

    case BasicJob::PING:
      WritePingMessage( clientSock);
      break;
    
    default:  // issue to job
      try {
        existing = m_jobManager.FindJob( header->id);
        existing->Command( header);
      } catch( ExceptionBase &) {
        LOG( "Job not found" << header->id);
      }
    }

    //return header into free ones
    m_headerPool.PutFreeItem( header);

  } catch( NetException &ne) {
    m_headerPool.PutFreeItem( header);
    LOG( "NetException in Server::EndPrimaryHeaderRead" << ne.what() );
  } catch( ExceptionBase &e) {
    m_headerPool.PutFreeItem( header);
    LOG( "ExceptionBase in Server::EndPrimaryHeaderRead" << e.what() );
  }

}

///////////////////////////////////////////////////////////////////////////////

void
Server::EndWritePingMessage( tcp::socket *clientSock,
        const boost::system::error_code& error)
{
  try {
    BasicSocket::HandleErrors( error);
  } catch( NetException &) {
    // nothing to do
  }

  delete clientSock;
}

///////////////////////////////////////////////////////////////////////////////

void
Server::WritePingMessage( tcp::socket *clientSock)
{
  // write ping message
  clientSock->async_write_some(
    boost::asio::buffer( m_pingMessage),
    boost::bind( &Server::EndWritePingMessage, this, clientSock,
      boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////////////


/** @} */

