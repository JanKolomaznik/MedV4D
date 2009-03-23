/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file netStream.h 
 * @{ 
 **/

#ifndef NETSTREAM_H
#define NETSTREAM_H

#define __USE_W32_SOCKETS     // for windows
#define _WIN32_WINNT 0x0501   // win ver

#include <boost/asio.hpp>
#include <boost/bind.hpp>

#include "common/mediumAccessor.h"
#include "common/Endianess.h"


namespace M4D
{
namespace RemoteComputing
{

/**
 *  Interface that is given to user of Imaging library. Through virtual
 *  function is actual implementation hidden in CellBE library.
 */
class NetAccessor : public IO::MediumAccessor
{
public:
	NetAccessor(boost::asio::ip::tcp::socket &socket);
	~NetAccessor();
	
	void PutData(const void *data, size_t length);
	void GetData(void *data, size_t length);
protected:
	boost::asio::ip::tcp::socket &m_socket_;
};

}
}

#endif

/** @} */

