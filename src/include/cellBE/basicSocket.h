#ifndef BASIC_SOCKET_H
#define BASIC_SOCKET_H


#define __USE_W32_SOCKETS     // for windows
#define _WIN32_WINNT 0x0501   // win ver

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <vector>

#include "netCommons.h"

namespace M4D
{
namespace CellBE
{

//////////////////////////////////////////////////
class NetException
  : public M4D::ErrorHandling::ExceptionBase
{
public :
  NetException( const std::string &s)
    : ExceptionBase( s) {}
};

//////////////////////////////////////////////////
/**
 *  Base class containing network functionality. Used BOOST::Asio for asynchronous networking and scatter gather.
 */
class BasicSocket
{
  friend class Server;

protected:
  boost::asio::ip::tcp::socket m_socket;

  BasicSocket(boost::asio::io_service &service) : m_socket(service) {}

  /**
   *  Unified handling of network errors.
   */
  static void HandleErrors( boost::system::error_code error)
  {
    if( error)
    {
      throw NetException( "Smth happend");
    }
  }

};

}
}
#endif

