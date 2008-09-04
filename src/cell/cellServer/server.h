/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file server.h 
 * @{ 
 **/

#ifndef SERVER_H
#define SERVER_H

#include <vector>
#include <map>

#include "serverJob.h"
#include "cellBE/resourcePool.h"
#include "jobManager.h"

namespace M4D
{
namespace CellBE
{

class Server
{

  public:
    Server(boost::asio::io_service& io_service);

  private:
    void Accept();

    void EndAccepted( boost::asio::ip::tcp::socket *clientSock,
        const boost::system::error_code& error);

    void EndPrimaryHeaderRead( boost::asio::ip::tcp::socket *clientSock,
        PrimaryJobHeader *header,
        const boost::system::error_code& error);

    // writes ping message. Address & server info
    void WritePingMessage( boost::asio::ip::tcp::socket *clientSock);
    void EndWritePingMessage( boost::asio::ip::tcp::socket *clientSock,
        const boost::system::error_code& error);

    boost::asio::ip::tcp::acceptor m_acceptor;

    JobManager m_jobManager;

    std::string m_pingMessage;

    NetStreamVector m_pingStream;

    static Pool<PrimaryJobHeader, 32> m_headerPool;
    
};

} // CellBE namespace
} // M4D namespace

#endif

/** @} */

