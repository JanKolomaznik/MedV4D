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

  void DeserializeFilterProperties( void);

  void BuildThePipeLine( void);  // TODO

  
  void ReadFilters( void);
  void EndFiltersRead( const boost::system::error_code& error);
  void EndDataSetPropertiesRead( const boost::system::error_code& error);
  
  /**
   *  If everything went right resulting dataSet is sent back to client
   *  along with responseHeader with OK result
   */
  void SendTheResultBack( void);
};

} // CellBE namespace
} // M4D namespace

#endif