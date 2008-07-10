#ifndef JOBBASE_H
#define JOBBASE_H

#include <vector>
#include "basicSocket.h"

#include "../Imaging/filterProperties.h"
#include "messageHeaders.h"
#include "resourcePool.h"


namespace M4D
{
namespace CellBE
{

class BasicJob
  : public BasicSocket
{
  friend class Server;

public:

  PrimaryJobHeader primHeader;
  SecondaryJobHeader secHeader;

protected:
  BasicJob(boost::asio::io_service &service);

  enum Action {
    CREATE,
    REEXEC,
    DESTROY,
    PING
  };

  M4D::Imaging::FilterVector m_filters;

  void EndSend( const boost::system::error_code& e);

  // pool for dataPeice headers
  static Pool< DataPieceHeader, 32> freeHeaders;

public:

  // data buffer to send
  //template< class T>
  struct DataBuff {
    void *data;
    size_t len;
  };

  // vector of databuffers to be send in one turn
  //template< class T>
  class DataBuffs : public std::vector< DataBuff >
  {
  };

  // Serialization
  //template<class T>
  void PutDataPiece( const DataBuffs &bufs);

  //template<class T>
  void PutDataPiece( const DataBuff &buf);

  // Deserialization
  //template<class T>
  void GetDataPiece( DataBuffs &bufs);

  //template<class T>
  void GetDataPiece( DataBuff &buf);

  // callback def
  typedef void (*JobCallback)(void);

  // events
  JobCallback onComplete;
  JobCallback onError;

};

} // CellBE namespace
} // M4D namespace

#endif

