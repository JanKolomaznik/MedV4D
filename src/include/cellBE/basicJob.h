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

  /**
   *  Definition jobs states
   */
  enum State {
    CREATION_FAILED,          // creating failed due to unrecognized filter
    FILTER_PROPS_OK,          // Updating filterProperties OK
    FILTER_PROPS_WRONG,       // Updating filterProperties failed
    DATASET_PROPS_OK,         // DataSet properties reading OK
    DATASET_PROPS_WRONG,      // Error while reading DS properties
    DATASET_OK,               // DS data recieved OK
    DATASET_WRONG,            // Error while recieving DS data
    EXECUTED,                 // remotePipe has successfuly executed
    ABORTED,                  //
    FAILED,                   //
    IDLE                      //
  };

  // callback def
  typedef void (*JobCallback)(void);

  // events
  JobCallback onComplete;
  JobCallback onError;

  // iPublicJob interface implementations
  void PutDataPiece( const DataBuffs &bufs);
  void PutDataPiece( const DataBuff &buf);
  
  NetStream * GetNetStream( void);

private:
  DataBuffs m_dataBufs;

protected:
  // ctor
  BasicJob( boost::asio::ip::tcp::socket *sock);

  /**
   *  Definition of basic action IDs.
   */
  enum Action {
    CREATE,       // request for job create
    EXEC,         // request for job execute
    DESTROY,      // request for job destroy
    DATASET,      // sending job's dataSet
    ABORT,        // sending abort req to abort computation
    FILTERS,      // sending job's filters settings
    PING          // ping message
  };

  virtual void OnDSRecieved( void) = 0;

  State m_state;

  static DataPieceHeader endHeader;   // data header saying noMoreData

  // filter setting vector
  FilterSerializerVector m_filters;
  
  // vector for sending & retrieving filterSettings & dataSetProperties data
  NetStreamVector filterSettingsSerialized;
  NetStreamVector m_dataSetPropsSerialized;

  void EndSend( const boost::system::error_code& e);

  // pool for dataPeice headers
  static Pool< DataPieceHeader, 32> freeHeaders;
  static Pool< ResponseHeader, 32> m_freeResponseHeaders;

  void GetDataPiece( DataBuffs &bufs, AbstractDataSetSerializer *dataSetSerializer);

  void ReadDataPeiceHeader( AbstractDataSetSerializer *dataSetSerializer);
  void EndReadDataPeiceHeader( const boost::system::error_code& error,
    DataPieceHeader *header, AbstractDataSetSerializer *dataSetSerializer);
  void EndReadDataPeice( const boost::system::error_code& error
    , AbstractDataSetSerializer *dataSetSerializer);

  // send EndingTag telling no more data will come
  void SendEndOfDataSetTag( void);

  AbstractDataSetSerializer *m_inDataSetSerializer;
  AbstractDataSetSerializer *m_outDataSetSerializer;

};

} // CellBE namespace
} // M4D namespace

#endif

