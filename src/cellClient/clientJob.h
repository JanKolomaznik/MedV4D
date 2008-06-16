#ifndef CLIENT_JOB_H
#define CLIENT_JOB_H

#include "cellBE/job.h"

namespace M4D
{
namespace CellBE
{

class ClientJob
  : public Job
{
  static uint32 lastID;

  NetStreamVector filterSettingsSerialized;
  void SerializeFiltersSetting();

  friend class ServerConnection;  // needs access to privates

  void Serialize( NetStream &s);
  void DeSerialize( NetStream &s);

public:
  ClientJob( bool isPersistent_);

  FilterSetting *GetFilter( std::string f);
  void AddFilter( std::string fID, FilterSetting *sett);

  // state
  enum State
  {
    Complete,
    Incomplete,
    Failed,
  };
  State state;

  // callback def
  typedef void (*JobCallback)(void);

  // events
  JobCallback onComplete;
  JobCallback onError;
};

}
}
#endif