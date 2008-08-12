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
  void Reexecute( void);
  void SendDestroy( void);

  void Serialize( NetStream &s);
  void DeSerialize( NetStream &s);

  void OnResponseRecieved( const boost::system::error_code& error
    , ResponseHeader *header);    

  // only CellClient can construct instances through CreateJob members
  ClientJob(
    FilterSerializerVector &filters
    , M4D::Imaging::AbstractDataSet *inDataSet
    , M4D::Imaging::AbstractDataSet *outdataSet
    , const std::string &address
    , boost::asio::io_service &service);

  ~ClientJob();

};

///////////////////////////////////////////////////////////////////////////////

}
}
#endif