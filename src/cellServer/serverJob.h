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
  std::vector<uint8> m_filterSettingContent;
  std::vector<FilterSetting *> filters;

  void DeserializeFilterSettings( void);

  void ReadSecondaryHeader( void);

  void EndSecondaryHeaderRead( const boost::system::error_code& error);
  void EndJobSettingsRead( const boost::system::error_code& error);

};

} // CellBE namespace
} // M4D namespace

#endif