#ifndef JOBBASE_H
#define JOBBASE_H

#include <vector>
#include <map>
//#include "Imaging/Image.h"
#include "filterProperties.h"
#include "messageHeaders.h"

namespace M4D
{
namespace CellBE
{

//template<typename elemType>
class Job
  : public iSerializable
{  
protected:
  //M4D::Imaging::Image<elemType, dimension> &image;

  friend class Server;

  typedef std::map<std::string, FilterSetting *> FilterMap;
  FilterMap m_filters;

  bool m_isPersistent;

  PrimaryJobHeader primHeader;
  SecondaryJobHeader secHeader;

  virtual void Serialize( NetStream &s) = 0;
  virtual void DeSerialize( NetStream &s) = 0;

public:
  uint32 jobID; 

  enum Action {
    CREATE,
    REEXEC,
    DESTROY
  };

  /**
  *  Job header, filters setting, data header preparation
  */

  /////////////////////////////////////////////////////////
  // Client methodes
  /////////////////////////////////////////////////////////
  // prepares job header for network transmission

};

} // CellBE namespace
} // M4D namespace

#endif