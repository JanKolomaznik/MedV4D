
#define __USE_W32_SOCKETS     // for windows
#define _WIN32_WINNT 0x0501   // win ver

#include <boost/bind.hpp>
#include <sstream>

#include "Common.h"
#include "cellBE/netCommons.h"
#include "server.h"

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
    boost::bind(&Server::EndAccepted, this, sock,
        boost::asio::placeholders::error) );
}

///////////////////////////////////////////////////////////////////////

void
Server::EndAccepted( tcp::socket *clientSock,
      const boost::system::error_code& error)
{
  if (!error)
  {
    ServerJob *job = new ServerJob();

    clientSock->async_read_some(
      boost::asio::buffer( (uint8*)&job->primHeader, sizeof(PrimaryJobHeader) ),
      boost::bind( &Server::EndPrimaryHeaderRead, this, clientSock, job,
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
Server::EndJobSettingsRead( tcp::socket *clientSock, ServerJob *job,
        const boost::system::error_code& error)
{
  if( error)
  {
    LOG( "onHeaderRead was called with error flag! ...");
    return;
  }
  else
  {
    // deserialize the filter settings
    {
      uint8 filterID;
      FilterSetting *fs;

      NetStreamArrayBuf s( &job->m_filterSettingContent[0], 
        job->m_filterSettingContent.size());
      try {
        while(1)  // it's breaked by exception when reading behind stream
        {
          s >> filterID;
          switch( (FilterID) filterID)
          {
          case Thresholding:
            fs = new ThresholdingSetting();
            fs->DeSerialize(s);
            job->filters.push_back( fs);
            break;

          default:
            LOG( "Unrecognized filter");
            throw ExceptionBase("Unrecognized filter");
          }
        }
      } catch( ExceptionBase) {
        // do nothing. Just continue
      }
    }

    // now split ways according action
    switch( (Job::Action) job->primHeader.action)
    {
    case Job::CREATE:
      // wait for data to arrive
      break;

    case Job::REEXEC:
      // wait for differrentiate data to arrive
      break;
    }

  }
}

///////////////////////////////////////////////////////////////////////

void
Server::EndPrimaryHeaderRead( tcp::socket *clientSock, ServerJob *job,
        const boost::system::error_code& error)
{
  // parse primary job header
  job->suppSerializer.DeSerializePrimMessHeader( &job->primHeader);

  ServerJob *existing;

  switch( job->primHeader.id)
  {
  case 0:
    // ping 'job'
    writePingMessage( clientSock);
    break;

  default:
    // regular job:
    switch( (Job::Action) job->primHeader.action)
    {
    case Job::CREATE:
      break;

    case Job::DESTROY:
      existing = findPersistendJob( job->primHeader.id);
      m_persistentJobs.erase( m_persistentJobs.find( job->primHeader.id) );
      // and delete existing because we are destroying it
      delete existing;
      delete job; // delete current one, because we dond need it anymore
      // TODO send ack?
      return;

    case Job::REEXEC:
      existing = findPersistendJob( job->primHeader.id);
      delete job; // delete current one, because we dond need it
      job = existing;
      break;

    default:
      LOG( "Unrecognized action job action. From: " << clientSock );
      throw ExceptionBase("Unrecognized action job action");
    }

    clientSock->async_read_some(
      boost::asio::buffer( (uint8*)&job->secHeader, sizeof( SecondaryJobHeader) ),
      boost::bind( &Server::EndSecondaryHeaderRead, this, clientSock, job,
        boost::asio::placeholders::error)
      );
  }
}

///////////////////////////////////////////////////////////////////////

void
Server::EndSecondaryHeaderRead( tcp::socket *clientSock, ServerJob *job,
        const boost::system::error_code& error)
{
  if( error)
  {
    LOG( "onHeaderRead was called with error flag! ...");
    return;
  }

  job->suppSerializer.DeSerializeSecMessHeader(&job->secHeader);

  // create Image, prepare data


  job->m_filterSettingContent.resize( job->secHeader.filterSettStreamLen);
  clientSock->async_read_some(
    boost::asio::buffer( job->m_filterSettingContent),
    boost::bind( &Server::EndJobSettingsRead, this, clientSock, job,
      boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////

void
Server::EndPingMessageWritten( tcp::socket *clientSock,
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
    boost::bind( &Server::EndPingMessageWritten, this, clientSock,
      boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////