#ifndef CLIENT_JOB_H
#define CLIENT_JOB_H

#include "cellBE/clientSocket.h"
#include "../Imaging/AbstractDataSet.h"

namespace M4D
{
namespace CellBE
{

class ClientJob
  : public ClientSocket
{
  friend class CellClient;  // needs access to privates

  static uint32 lastID;
  void GenerateJobID( void);

  enum InnerState
  {
    CREATED,  // after constructor
    DATASET_SPECIFIED,  // after 
    SUBMITED, // after
    SENT,
    AWAITING_REQUEST,
    IDLE
  };

  NetStreamVector filterSettingsSerialized;
  void SerializeFiltersSetting( void);

  NetStreamVector m_dataSetPropsSerialized;

  void SendHeaders( void);

  void Serialize( NetStream &s);
  void DeSerialize( NetStream &s);

  // only CellClient can construct instances through CreateJob members
  ClientJob(
    M4D::Imaging::FilterVector &filters
    , M4D::Imaging::AbstractDataSet *dataSet
    , const std::string &address
    , boost::asio::io_service &service);

  M4D::Imaging::AbstractDataSet *m_dataSet;

public:

  // state
  enum State
  {
    Complete,
    Incomplete,
    Failed,
  };
  State state;

};

///////////////////////////////////////////////////////////////////////

}
}
#endif