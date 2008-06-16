#ifndef SERVER_H
#define SERVER_H

#define __USE_W32_SOCKETS     // for windows
#define _WIN32_WINNT 0x0501   // win ver

#include <vector>
#include <map>
#include <boost/asio.hpp>

#include "serverJob.h"

namespace M4D
{
namespace CellBE
{
  using boost::asio::ip::tcp;

  class Server
  {
#define PING_MESSAGE_CONTENT "Hi.This is cell server."

  public:
    Server(boost::asio::io_service& io_service, uint16 port);

  private:
    void Accept();

    void EndAccepted( tcp::socket *clientSock,
        const boost::system::error_code& error);

    void EndPrimaryHeaderRead( tcp::socket *clientSock, ServerJob *job,
        const boost::system::error_code& error);

    void EndSecondaryHeaderRead( tcp::socket *clientSock, ServerJob *job,
        const boost::system::error_code& error);

    void EndJobSettingsRead( tcp::socket *clientSock, ServerJob *job,
        const boost::system::error_code& error);

    void EndPingMessageWritten( tcp::socket *clientSock,
        const boost::system::error_code& error);

    // writes ping message. Address & server info
    void writePingMessage( tcp::socket *clientSock);

    tcp::acceptor m_acceptor;

    /*typedef vector<boost::asio::ip::tcp::socket> SockPool;
    SockPool m_socketPool;*/
    std::string m_pingMessage;

    typedef std::map<uint32, ServerJob *> JobMap;
    JobMap m_persistentJobs;

    inline ServerJob *findPersistendJob( uint32 id)
    {
      JobMap::iterator it = m_persistentJobs.find(id);
      if( it == m_persistentJobs.end() )
        throw ExceptionBase("Job not found");
      else
        return it->second;
    }
    
  };

} // CellBE namespace
} // M4D namespace

#endif
