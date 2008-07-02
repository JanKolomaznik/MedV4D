#ifndef JOBBASE_H
#define JOBBASE_H

#include <vector>
#include "filterProperties.h"
#include "messageHeaders.h"
#include "dataSetProperties.h"
#include "resourcePool.h"

#include "basicSocket.h"

namespace M4D
{
namespace CellBE
{

class BasicJob
  : public BasicSocket
{
  friend class Server;

public:
  typedef std::vector<FilterSetting *> FilterVector;

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

  FilterVector m_filters;

  void EndSend( const boost::system::error_code& e);

  // pool for dataPeice headers //////////////////////////////////
  Pool< DataPieceHeader, 32> freeHeaders;
//#define MAX_PIECE_HEADERS 32
//  static DataPieceHeader p_pieceHeaders[MAX_PIECE_HEADERS];
//  typedef std::vector< DataPieceHeader *> FreeDataPieceVect;
//  static FreeDataPieceVect p_freeDPHeaders;
//
//  static void InitDataPieceHeaders( void)
//  {
//    for( int i=0; i<MAX_PIECE_HEADERS; i++)
//      p_freeDPHeaders.push_back( &p_pieceHeaders[i]);
//  }
  ////////////////////////////////////////////////////////////////

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