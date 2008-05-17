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
  ServerConnection( const std::string &address, boost::asio::io_service &service);

  inline const std::string Address(void) { return m_address; }

  void SendJob( Job &job);

private: 
  boost::asio::ip::tcp::socket m_socket;
  std::string m_address;

  boost::array<char, 8> m_pok;

  void Connect( boost::asio::io_service &service);

  void OnJobWritten( const boost::system::error_code& e, Job &j);

  void OnJobResponseHeaderRead( const boost::system::error_code& e,
    const size_t bytesRead, Job &j);

  void OnJobResponseBodyRead( const boost::system::error_code& e,
    const size_t bytesRead, Job &j);
};

} // CellBE namespace
} // M4D namespace

#endif