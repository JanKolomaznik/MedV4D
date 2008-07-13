#ifndef JOBBASE_H
#define JOBBASE_H

#include <vector>
#include "basicSocket.h"

#include "../Imaging/filterProperties.h"
#include "messageHeaders.h"
#include "resourcePool.h"
#include "iPublicJob.h"


namespace M4D
{
namespace CellBE
{
/**
 *  Base class for job. Contains common parts for client side even server side. 
 */
class BasicJob
  : public BasicSocket, public iPublicJob
{
  friend class Server;

public:

  // header struct used for sending headers
  PrimaryJobHeader primHeader;
  SecondaryJobHeader secHeader;

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
  M4D::Imaging::FilterVector m_filters;

  void EndSend( const boost::system::error_code& e);

  // pool for dataPeice headers
  static Pool< DataPieceHeader, 32> freeHeaders;
  
public:
  // callback def
  typedef void (*JobCallback)(void);

  // events
  JobCallback onComplete;
  JobCallback onError;

  void PutDataPiece( const DataBuffs &bufs);
  void PutDataPiece( const DataBuff &buf);
  void GetDataPiece( DataBuffs &bufs);
  void GetDataPiece( DataBuff &buf);

};

} // CellBE namespace
} // M4D namespace

#endif

