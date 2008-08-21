#ifndef CLIENT_JOB_H
#define CLIENT_JOB_H

#include "cellBE/clientSocket.h"

namespace M4D
{
namespace CellBE
{

/**
 *  Client side job class. Advances functionality for ID generation.
 *  Has some supporting members that incomming and outcomming data
 *  are store while assynchronous operation are in progress.
 */
class ClientJob
  : public ClientSocket
{
  friend class CellClient;  // needs access to privates

  static uint32 lastID;
  void GenerateJobID( void);

  NetStreamVector m_remotePipeDefSerialized;

  void SerializeFiltersProperties( void);
  void SerializeRemotePipeDefinition( void);

  void Serialize( NetStream &s);
  void DeSerialize( NetStream &s);

  void OnResponseRecieved( const boost::system::error_code& error
    , ResponseHeader *header);

  // only CellClient can construct instances through CreateJob members
  ClientJob(
    FilterSerializerVector &filters
    //, AbstractDataSetSerializer *inDataSetSeralizer
    //, AbstractDataSetSerializer *outDataSetSerializer
    , const std::string &address
    , boost::asio::io_service &service);

  AbstractDataSetSerializer *m_inDataSetSerializer;
  AbstractDataSetSerializer *m_outDataSetSerializer;

public:
  ~ClientJob();

  void SendCreate( void);
  void SendDataSetProps( void);
  void SendDataSet( void);
  void SendFilterProperties( void);
  void SendDestroy( void);

  void SetDataSets( const M4D::Imaging::AbstractDataSet &inDataSet
                  , M4D::Imaging::AbstractDataSet &outdataSet);

};

///////////////////////////////////////////////////////////////////////////////

}
}
#endif