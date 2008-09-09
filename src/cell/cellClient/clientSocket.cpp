/**
 *  @ingroup cellbe
 *  @file clientSocket.cpp
 *  @author Vaclav Klecanda
 */

#include <string>
#include <sstream>
#include <vector>

#include <iostream>

#include "Common.h"
#include "cellBE/netCommons.h"
#include "cellBE/clientSocket.h"

using namespace std;
using boost::asio::ip::tcp;

namespace M4D {
namespace CellBE {

///////////////////////////////////////////////////////////////////////////////

ClientSocket::ClientSocket( const std::string &address,
  boost::asio::io_service &service)
  : BasicJob( new tcp::socket( service)), m_address(address)
{
  Connect( service);
}

///////////////////////////////////////////////////////////////////////////////

void
ClientSocket::Connect( boost::asio::io_service &service)
{
  tcp::resolver resolver(service);

  stringstream port;
  port << SERVER_PORT;

  tcp::resolver::query query(
    m_address,
    port.str(),
    tcp::resolver::query::numeric_service);

  tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
  tcp::resolver::iterator end;

  boost::system::error_code error = boost::asio::error::host_not_found;
  while (error && endpoint_iterator != end)
  {
    m_socket->close();
    m_socket->connect(*endpoint_iterator++, error);
  }
  if (error)
    throw M4D::ErrorHandling::ExceptionBase(
      "Not able to connect to Cell sever");
}

///////////////////////////////////////////////////////////////////////////////

}
}