
#include "common/Common.h"
#include "remoteComp/netAccessor.h"
#include "../netCommons.h"

using namespace M4D::RemoteComputing;
using namespace std;

/////////////////////////////////////////////////////////////////////////////

NetAccessor::NetAccessor(asio::ip::tcp::socket &socket)
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
//		asio::write( m_socket_,
//			asio::buffer( data, length), ec );
//	size_t written = m_socket_.write_some( 
//			asio::buffer( data, length), ec );
//	if(ec || (written != length) )
//		throw NetException();
	try {
		//size_t written = 
			asio::write(
					m_socket_, asio::buffer(data, length));
	} catch (asio::system_error &e) {
		if(e.code() == asio::error::eof )
			throw DisconnectedException();
		else
			throw NetException();
	}
}

/////////////////////////////////////////////////////////////////////////////
void
NetAccessor::GetData(void *data, size_t length)	
{
	asio::read( m_socket_, asio::buffer(data, length));
}
/////////////////////////////////////////////////////////////////////////////
