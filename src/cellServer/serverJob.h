#ifndef SERVERJOB_H
#define SERVERJOB_H

#include "cellBE/basicJob.h"
#include "Imaging/PipelineContainer.h"

namespace M4D
{
namespace CellBE
{

class ServerJob
  : public BasicJob
{
  friend class Server;
  
private:
  ServerJob(boost::asio::io_service &service) : BasicJob(service) {}

  std::vector<uint8> m_filterSettingContent;

  M4D::Imaging::PipelineContainer m_pipeLine;

  M4D::Imaging::AbstractPipeFilter *m_pipelineBegin, *m_pipelineEnd;

  void DeserializeFilterPropertiesAndBuildPipeline( void);
  
  void ReadFilters( void);
  void EndFiltersRead( const boost::system::error_code& error);
  void EndDataSetPropertiesRead( const boost::system::error_code& error);
  
  /**
   *  Sends result message back to client. Within the method is switch
   *  according result param. If OK, resulting dataSet is then send.
   *  Else just apropriate ResponseID is sent defining the error with
   *  no other data on the tail. Also resultPropertiesLen item of ResultHeader
   *  is not used.
   */
  void SendResultBack( ResponseID result);
  void OnResultHeaderSent( const boost::system::error_code& error
    , ResponseHeader *h);
};

} // CellBE namespace
} // M4D namespace

#endif