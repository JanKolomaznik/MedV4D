#ifndef SERVERJOB_H
#define SERVERJOB_H

#include "cellBE/job.h"

namespace M4D
{
namespace CellBE
{

//template<typename elemType, uint8 dimension>
class ServerJob
  : public Job
{
  friend class Server;
  
private:
  std::vector<uint8> m_filterSettingContent;
  std::vector<FilterSetting *> filters;

  void Serialize( NetStream &s);
  void DeSerialize( NetStream &s);
};

} // CellBE namespace
} // M4D namespace

#endif