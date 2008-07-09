
#include <vector>
#include "Common.h"
#include "cellBE/basicJob.h"

using namespace M4D::CellBE;
using namespace std;

Pool< DataPieceHeader, 32> BasicJob::freeHeaders;

///////////////////////////////////////////////////////////////////////

BasicJob::BasicJob(boost::asio::io_service &service)  
    : BasicSocket(service)
    , onComplete( NULL)
    , onError(NULL)
{
}

///////////////////////////////////////////////////////////////////////

void
BasicJob::PutDataPiece( const M4D::CellBE::BasicJob::DataBuff &buf)
{
  DataBuffs bufs;
  bufs.push_back( buf);
  PutDataPiece( bufs);
}

///////////////////////////////////////////////////////////////////////

void
BasicJob::PutDataPiece( const M4D::CellBE::BasicJob::DataBuffs &bufs)
{
  // count total length of all buffers
  size_t totalLen = 0;

  DataPieceHeader *header = freeHeaders.GetFreeItem();

  vector<boost::asio::const_buffer> buffers;

  // push header of all dataPiece
  buffers.push_back( boost::asio::buffer( header, sizeof( DataPieceHeader)) );

  // push rest of bufs
  const DataBuff *buf;
  for( DataBuffs::const_iterator it=bufs.begin(); it != bufs.end(); it++)
  {
    buf = (DataBuff*) & it;
    buffers.push_back( boost::asio::const_buffer( buf->data, buf->len) );
    totalLen += buf->len;
  }

  header->pieceSize = (uint32) totalLen;
  DataPieceHeader::Serialize( header);

  // get free dataPieceHeader
  m_socket.async_write_some(
    buffers, 
    boost::bind( & BasicJob::EndSend, this, boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////

void 
BasicJob::EndSend( const boost::system::error_code& e)
{
  try {
    HandleErrors( e);
  } catch( ExceptionBase &) {
    if( onError != NULL)
      onError();
  }
}

///////////////////////////////////////////////////////////////////////