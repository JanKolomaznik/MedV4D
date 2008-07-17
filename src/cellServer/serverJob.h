#ifndef SERVERJOB_H
#define SERVERJOB_H

#include "cellBE/basicJob.h"
#include "Imaging/AbstractDataSet.h"

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

  M4D::Imaging::AbstractDataSet *dataSet;

  void DeserializeFilterSettings( void);

  void BuildThePipeLine( void);  // TODO
  void CreateDataSet( void);     // TODO

  void ReadSecondaryHeader( void);
  void ReadDataPeiceHeader( void);

  void EndSecondaryHeaderRead( const boost::system::error_code& error);
  void EndJobSettingsRead( const boost::system::error_code& error);
  void EndDataSetPropertiesRead( const boost::system::error_code& error);
  void EndReadDataPeiceHeader( const boost::system::error_code& error,
    DataPieceHeader *header);

};

} // CellBE namespace
} // M4D namespace

#endif