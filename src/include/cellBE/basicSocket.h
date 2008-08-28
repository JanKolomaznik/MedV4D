/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file basicSocket.h 
 * @{ 
 **/

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

/**
 *  Base class containing network functionality. Used BOOST::Asio for asynchronous networking and scatter gather.
 */
class BasicSocket
{
  friend class Server;

protected:
  boost::asio::ip::tcp::socket *m_socket;

  BasicSocket(boost::asio::ip::tcp::socket *sock) : m_socket(sock) {}

  /**
   *  Unified handling of network errors.
   */
  static void HandleErrors( boost::system::error_code error)
  {
    if( error)
    {
      std::stringstream s;
      s << "Smth happend on network in HandleErrors (err:" << error << ")";
      throw NetException( s.str() );
    }
  }

};

}
}
#endif


/** @} */

