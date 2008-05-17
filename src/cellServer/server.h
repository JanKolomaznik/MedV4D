#ifndef SERVER_H
#define SERVER_H

#define __USE_W32_SOCKETS     // for windows
#define _WIN32_WINNT 0x0501   // win ver

#include <vector>
#include <boost/asio.hpp>

#include "cellBE/job.h"

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

    void onAccepted( tcp::socket *clientSock,
        const boost::system::error_code& error);

    void onHeaderRead( tcp::socket *clientSock, MessageHeader *header,
        const boost::system::error_code& error);

    void onPingMessageWritten( tcp::socket *clientSock,
        const boost::system::error_code& error);

    // writes ping message. Address & server info
    void writePingMessage( tcp::socket *clientSock);

    tcp::acceptor m_acceptor;

    /*typedef vector<boost::asio::ip::tcp::socket> SockPool;
    SockPool m_socketPool;*/
    std::string m_pingMessage;

    char inbound[MESSLEN];
  };

} // CellBE namespace
} // M4D namespace

#endif
