#ifndef SERVERCONN_HPP
#define SERVERCONN_HPP

#define __USE_W32_SOCKETS     // for windows
#define _WIN32_WINNT 0x0501   // win ver

#include <boost/asio.hpp>

#include "clientJob.h"

namespace M4D
{
namespace CellBE
{
  using boost::asio::ip::tcp;

class ServerConnection
{
public:
  ServerConnection( const std::string &address, boost::asio::io_service &service);

  void SendJob( ClientJob *job);
  void Resend( ClientJob *job);
  void QuitJob( ClientJob *job);

private: 
  boost::asio::ip::tcp::socket m_socket;
  std::string m_address;

  //////////////////////////////////////////////
  void SendData( ClientJob *j);
  void ReadResponseHeader( ClientJob *job);
  //////////////////////////////////////////////

  void Connect( boost::asio::io_service &service);

  // send callbacks
  void EndSendJobHeader( const boost::system::error_code& e, ClientJob *j);
  void EndSendData( const boost::system::error_code& e, ClientJob *j);

  void OnJobResponseHeaderRead( const boost::system::error_code& e,
    const size_t bytesRead, ClientJob *j);

  void OnJobResponseBodyRead( const boost::system::error_code& e,
    const size_t bytesRead, ClientJob *j);

};

} // CellBE namespace
} // M4D namespace

#endif