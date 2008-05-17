
#define __USE_W32_SOCKETS     // for windows
#define _WIN32_WINNT 0x0501   // win ver

#include <boost/bind.hpp>
#include <sstream>

#include "Common.h"
#include "cellBE/commDefs.h"
#include "server.h"

#include <boost/archive/text_iarchive.hpp>

using namespace M4D::CellBE;

///////////////////////////////////////////////////////////////////////

Server::Server(boost::asio::io_service &io_service, uint16 port)
  : m_acceptor(io_service, tcp::endpoint(tcp::v4(), port))
{
  // prepare ping message content
  {
    /*std::stringstream tmp;
    tmp << PING_MESSAGE_CONTENT;
    tmp << "(IP: " << m_acceptor.local_endpoint().address() << ", port:";
    tmp << m_acceptor.local_endpoint().port() << ")";

    uint8_stream buf;
    Serializer::ToStream( buf, (uint16) Mess_Ping);
    Serializer::ToStream( buf, (uint16) tmp.str().length());
    buf << tmp;*/

  }

  // start server accepting
  Accept();
}

///////////////////////////////////////////////////////////////////////

void
Server::Accept( void)
{
  // create new instance of OneClientConnection
  //OneClientConnection *newConnection = 
  //  new OneClientConnection(m_acceptor.io_service());
  tcp::socket *sock = new tcp::socket(m_acceptor.io_service());

  // and start accepting
  m_acceptor.async_accept(
    *sock,
    boost::bind(&Server::onAccepted, this, sock,
        boost::asio::placeholders::error) );
}

///////////////////////////////////////////////////////////////////////

void
Server::onAccepted( tcp::socket *clientSock,
      const boost::system::error_code& error)
{
  if (!error)
  {
    MessageHeader *header = new MessageHeader();

    clientSock->async_read_some(
      boost::asio::buffer(inbound, Job::messLen),
      boost::bind( &Server::onHeaderRead, this, clientSock, header,
        boost::asio::placeholders::error)
      );

    LOG( "Accepted conn from:" 
      << clientSock->remote_endpoint().address() );      
  }
  else
  {
    LOG("On Accepted called with error!");
  }

  // accept again
  Accept();
}

///////////////////////////////////////////////////////////////////////

void
Server::onPingMessageWritten( tcp::socket *clientSock,
        const boost::system::error_code& error)
{
  if( error)
  {
    LOG( "Ping Message Sending Failed! ... ");
  }

  delete clientSock;
}

///////////////////////////////////////////////////////////////////////

void
Server::writePingMessage( tcp::socket *clientSock)
{
  // write ping message
  clientSock->async_write_some(
    boost::asio::buffer( m_pingMessage),
    boost::bind( &Server::onPingMessageWritten, this, clientSock,
      boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////

void
Server::onHeaderRead( tcp::socket *clientSock, MessageHeader *header,
        const boost::system::error_code& error)
{
  if( error)
  {
    LOG( "onHeaderRead was called with error flag! ...");
    return;
  }

  //std::istringstream headerStream(header->data);
  std::string inboundStr( inbound);
  std::stringstream strStram( inboundStr);

  boost::archive::text_iarchive archive(strStram);

  Job j;

  archive >> j;

  uint16 messID = 0;
  //headerStream >> messID;

  switch( messID)
  {
  case Mess_Ping:
    writePingMessage(clientSock);
    break;
  case Mess_Job:
    // start special thread that will do the job?
    break;
  }
}