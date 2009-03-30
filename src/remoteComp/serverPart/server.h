/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file server.h 
 * @{ 
 **/

#ifndef SERVER_H
#define SERVER_H

#include "Imaging/AbstractFilter.h"
#include "Imaging/PipelineContainer.h"
#include "remoteFilterFactory.h"
#include "../netAccessor.h"

namespace M4D
{
namespace RemoteComputing
{

class Server
{

  public:
    Server(asio::io_service& io_service);
    
    void OnExecutionDone( void);
    void OnExecutionFailed( void);

  private:
    void Accept();
    
    void ReadCommand(void);
    
    void OnClientDisconnected();
    
    void CreatePipeline();
    void ReadDataSet();
    void ReadFilterProperties();

    asio::ip::tcp::acceptor m_acceptor;
    asio::ip::tcp::socket m_socket_;
    
    NetAccessor netAccessor_;
    
    // pointer to currently only one filter
    M4D::Imaging::AbstractPipeFilter *m_filter;
    iRemoteFilterProperties *m_props;
    M4D::Imaging::ConnectionInterface *m_connWithOutputDataSet;
    M4D::Imaging::PipelineContainer m_pipeLine;
};

} // CellBE namespace
} // M4D namespace

#endif

/** @} */

