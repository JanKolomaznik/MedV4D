
#include <sstream>

#include "Common.h"
#include "server.h"

using namespace M4D::CellBE;

#ifdef MESSAGE_HEADERS_H
int i=10;
#else
int i=20;
#endif

///////////////////////////////////////////////////////////////////////

Server::Server(boost::asio::io_service &io_service)
  : m_acceptor(io_service, tcp::endpoint(tcp::v4(), (uint16) SERVER_PORT))
{
  // prepare ping message content
#define PING_MESSAGE_CONTENT "Hi.This is cell server."
  {
    NetStreamVect s;
    s << PING_MESSAGE_CONTENT;
    s << "(IP: " << m_acceptor.local_endpoint().address() << ", port:";
    s << m_acceptor.local_endpoint().port() << ")";

    m_pingStream << (uint16) s.length();
    m_pingStream << s;
  }

  // free headers are all
  for( uint32 i=0; i < HEADER_POOL_SIZE; i++)
    m_freeHeaders.push_back( &m_freeHeaders[i]);

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
    boost::bind(&Server::EndAccepted, this, sock,
        boost::asio::placeholders::error) );
}

///////////////////////////////////////////////////////////////////////

void
Server::EndAccepted( tcp::socket *clientSock,
      const boost::system::error_code& error)
{
  try {

    HandleErrors( error);

    PrimaryJobHeader *freeHeader = m_freeHeaders.back();
    m_freeHeaders.pop_back();

    clientSock->async_read_some(
      boost::asio::buffer( (uint8*)freeHeader, sizeof(PrimaryJobHeader) ),
      boost::bind( &Server::EndPrimaryHeaderRead, this, clientSock, freeHeader,
        boost::asio::placeholders::error)
      );

    LOG( "Accepted conn from:" 
      << clientSock->remote_endpoint().address() );

  } catch ( NetException e) {
    LOG("NetException");
  }

  // accept again
  Accept();
}

///////////////////////////////////////////////////////////////////////

void
Server::EndPrimaryHeaderRead( tcp::socket *clientSock, PrimaryJobHeader *header,
        const boost::system::error_code& error)
{
  try {
    HandleErrors( error);

    // parse primary job header
    PrimaryJobHeader::Deserialize( header);

    ServerJob *existing;

    switch( (Job::Action) header->action)
    {
    case Job::CREATE:
      existing = new ServerJob();
      existing->primHeader.id = header->id;
      existing->primHeader.endian = header->endian;
      existing->primHeader.dataSetType = header->dataSetType;

      m_jobManager.add( existing);

      existing->ReadSecondaryHeader();
      break;

    case Job::REEXEC:
      existing = m_jobManager.find( header.id);
      existing->ReadSecondaryHeader();
      break;

    case Job::DESTROY:
      try {
        existing = m_jobManager.find( header.id);
      } catch( ExceptionBase &e) {
        LOG( "Job not found" << header.id);
      }
      m_jobManager.remove( header.id );

      // TODO send ack?
      return;

    case Job::PING:
      WritePingMessage( clientSock);
      break;

    default:
      LOG( "Unrecognized action job action. From: " << clientSock );
      throw ExceptionBase("Unrecognized action job action");
    }

    //return header into free ones
    m_freeHeaders.push_back( header);

  }catch( ExceptionBase &) {
  }

}

///////////////////////////////////////////////////////////////////////

void
Server::EndWritePingMessage( tcp::socket *clientSock,
        const boost::system::error_code& error)
{
  try {
    HandleErrors( error);
  } catch( NetException &) {
    // nothing to do
  }

  delete clientSock;
}

///////////////////////////////////////////////////////////////////////

void
Server::WritePingMessage( tcp::socket *clientSock)
{
  // write ping message
  clientSock->async_write_some(
    boost::asio::buffer( m_pingMessage),
    boost::bind( &Server::EndPingMessageWritten, this, clientSock,
      boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////