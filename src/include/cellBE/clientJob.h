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

  // nothing to do. Everything is done automaticaly after 
  // RemoteFilter::ProccessImage ends
  void OnDSRecieved( void) {}

  void SerializeFiltersProperties( void);
  void SerializeRemotePipeDefinition( void);

  void ProcessResponse( const ResponseHeader &header);

  void ReadResultingDataSet( void);

  // only CellClient can construct instances through CreateJob members
  ClientJob(
    FilterSerializerVector &filters
    , const std::string &address
    , boost::asio::io_service &service);

public:
  ~ClientJob();

  void SendCreate( void);
  void SendDataSet( void);
  void SendFilterProperties( void);
  void SendExecute( void);
  void SendDestroy( void);

  void SetDataSets( const M4D::Imaging::AbstractDataSet &inDataSet
                  , M4D::Imaging::AbstractDataSet &outdataSet);

};

///////////////////////////////////////////////////////////////////////////////

class WrongJobStateException
  : public ExceptionBase
{
};

}
}
#endif