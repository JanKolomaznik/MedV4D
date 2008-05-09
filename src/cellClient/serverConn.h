#ifndef SERVERCONN_HPP
#define SERVERCONN_HPP

#define __USE_W32_SOCKETS     // for windows
#define _WIN32_WINNT 0x0501   // win ver

#include <boost/asio.hpp>

#include "cellBE/job.h"

namespace M4D
{
namespace CellBE
{
  using boost::asio::ip::tcp;

class ServerConnection
{
public:
  ServerConnection( const std::string &address);

  inline const std::string Address(void) { return m_address; }
  inline bool IsConnected(void) { return m_connected; }
  void Connect( void);

  void SendJob( Job &job);

private:
  
  boost::asio::io_service m_io_service;

  boost::asio::ip::tcp::socket *m_socket;
  std::string m_address;
  bool m_connected;

  void OnJobWritten( const boost::system::error_code& e,
    const uint16 bytesWritten, const Job &j);

  void OnJobResponseHeaderRead( const boost::system::error_code& e,
    const uint16 bytesWritten, Job &j);
  void OnJobResponseBodyRead( const boost::system::error_code& e,
    const uint16 bytesWritten, Job &j);
};

} // CellBE namespace
} // M4D namespace

#endif