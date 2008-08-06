#ifndef JOBBASE_H
#define JOBBASE_H

#include <vector>
#include "basicSocket.h"

#include "../Imaging/filterProperties.h"
#include "messageHeaders.h"
#include "resourcePool.h"
#include "iPublicJob.h"
#include "AbstractDataSetSerializer.h"
#include "AbstractFilterSerializer.h"


namespace M4D
{
namespace CellBE
{
/**
 *  Base class for job. Contains common parts for client side and server side.
 */
class BasicJob
  : public BasicSocket, public iPublicJob
{
  friend class Server;

public:

  // header struct used for sending headers
  PrimaryJobHeader primHeader;

protected:
  // ctor
  BasicJob(boost::asio::io_service &service);

  /**
   *  Definition of basic action IDs.
   */
  enum Action {
    CREATE,
    REEXEC,
    DESTROY,
    PING
  };

  // filter setting vector
  FilterPropsVector m_filters;

  void EndSend( const boost::system::error_code& e);

  // pool for dataPeice headers
  static Pool< DataPieceHeader, 32> freeHeaders;
  
public:
  // callback def
  typedef void (*JobCallback)(void);

  // events
  JobCallback onComplete;
  JobCallback onError;

  // iPublicJob interface implementations
  void PutDataPiece( const DataBuffs &bufs);
  void PutDataPiece( const DataBuff &buf);
  void GetDataPiece( DataBuffs &bufs);
  void GetDataPiece( DataBuff &buf);
  NetStream * GetNetStream( void);

};

} // CellBE namespace
} // M4D namespace

#endif

