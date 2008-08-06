#ifndef JOBBASE_H
#define JOBBASE_H

#include "basicSocket.h"

#include "messageHeaders.h"
#include "resourcePool.h"
#include "iPublicJob.h"
#include "GeneralDataSetSerializer.h"
#include "GeneralFilterSerializer.h"

#include "Imaging/AbstractDataSet.h"


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

private:
  DataBuffs m_dataBufs;

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

  static DataPieceHeader endHeader;   // data header saying noMoreData

  // filter setting vector
  FilterPropsVector m_filters;
  
  // vector for sending & retrieving filterSettings & dataSetProperties data
  NetStreamVector filterSettingsSerialized;
  NetStreamVector m_dataSetPropsSerialized;

  void EndSend( const boost::system::error_code& e);

  // pool for dataPeice headers
  static Pool< DataPieceHeader, 32> freeHeaders;

  void GetDataPiece( DataBuffs &bufs, AbstractDataSetSerializer *dataSetSerializer);

  void ReadDataPeiceHeader( AbstractDataSetSerializer *dataSetSerializer);
  void EndReadDataPeiceHeader( const boost::system::error_code& error,
    DataPieceHeader *header, AbstractDataSetSerializer *dataSetSerializer);
  void EndReadDataPeice( const boost::system::error_code& error
    , AbstractDataSetSerializer *dataSetSerializer);

  
public:
  // callback def
  typedef void (*JobCallback)(void);

  // events
  JobCallback onComplete;
  JobCallback onError;

  // iPublicJob interface implementations
  void PutDataPiece( const DataBuffs &bufs);
  void PutDataPiece( const DataBuff &buf);
  
  NetStream * GetNetStream( void);

};

} // CellBE namespace
} // M4D namespace

#endif

