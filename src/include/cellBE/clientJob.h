#ifndef CLIENT_JOB_H
#define CLIENT_JOB_H

#include "cellBE/clientSocket.h"

namespace M4D
{
namespace CellBE
{

/**
 *  Client side job class. Advances functionality for ID generation.
 */
class ClientJob
  : public ClientSocket
{
  friend class CellClient;  // needs access to privates

  static uint32 lastID;
  void GenerateJobID( void);

  void SerializeFiltersProperties( void);

  void SendCreate( void);

  void Serialize( NetStream &s);
  void DeSerialize( NetStream &s);

  // only CellClient can construct instances through CreateJob members
  ClientJob(
    FilterPropsVector &filters
    , M4D::Imaging::AbstractDataSet *dataSet
    , const std::string &address
    , boost::asio::io_service &service);

  M4D::Imaging::AbstractDataSet *m_dataSet;

public:

};

///////////////////////////////////////////////////////////////////////

}
}
#endif