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
    Server(boost::asio::io_service& io_service);
    
    void OnExecutionDone( void);
    void OnExecutionFailed( void);

  private:
    void Accept();

    void EndAccepted( const boost::system::error_code& error);
    
    void ReadCommand(void);
    
    void CreatePipeline();
    void ReadDataSet();
    void ReadFilterProperties();

    boost::asio::ip::tcp::acceptor m_acceptor;
    boost::asio::ip::tcp::socket m_socket_;
    
    NetAccessor netAccessor_;
    
    // pointers to first & last filter in pipeline
    M4D::Imaging::AbstractPipeFilter *m_pipelineBegin, *m_pipelineEnd;
    M4D::Imaging::PipelineContainer m_pipeLine;
    
    void CleenupPipeline(void);
    
    void HandleErrors(const boost::system::error_code& error);
};

} // CellBE namespace
} // M4D namespace

#endif

/** @} */

