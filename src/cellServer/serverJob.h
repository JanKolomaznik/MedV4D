#ifndef SERVERJOB_H
#define SERVERJOB_H

#include "cellBE/basicJob.h"

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

  void DeserializeFilterSettings( void);

  void ReadSecondaryHeader( void);
  void ReadDataPeiceHeader( void);

  void EndSecondaryHeaderRead( const boost::system::error_code& error);
  void EndJobSettingsRead( const boost::system::error_code& error);
  void EndReadDataPeiceHeader( const boost::system::error_code& error,
    DataPieceHeader *header);

};

} // CellBE namespace
} // M4D namespace

#endif