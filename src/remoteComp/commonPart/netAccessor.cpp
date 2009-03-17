
#include "Common.h"
#include "remoteComp/netAccessor.h"
#include "../netCommons.h"

using namespace M4D::RemoteComputing;
using namespace std;

/////////////////////////////////////////////////////////////////////////////

NetAccessor::NetAccessor(boost::asio::ip::tcp::socket &socket)
	: m_socket_( socket)
{
}
/////////////////////////////////////////////////////////////////////////////
NetAccessor::~NetAccessor()
{
}

/////////////////////////////////////////////////////////////////////////////
void
NetAccessor::PutData(const void *data, size_t length)
{
//	boost::system::error_code ec;
//		boost::asio::write( m_socket_,
//			boost::asio::buffer( data, length), ec );
//	size_t written = m_socket_.write_some( 
//			boost::asio::buffer( data, length), ec );
//	if(ec || (written != length) )
//		throw NetException();
	try {
		//size_t written = 
			boost::asio::write(
					m_socket_, boost::asio::buffer(data, length));
	} catch (boost::system::system_error &e) {
		if(e.code() == boost::asio::error::eof )
			throw DisconnectedException();
		else
			throw NetException();
	}
}

/////////////////////////////////////////////////////////////////////////////
void
NetAccessor::GetData(void *data, size_t length)	
{
	boost::asio::read( m_socket_, boost::asio::buffer(data, length));
}
/////////////////////////////////////////////////////////////////////////////
