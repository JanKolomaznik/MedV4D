
#include <vector>
#include "Common.h"
#include "cellBE/basicJob.h"

using namespace M4D::CellBE;
using namespace std;

Pool< DataPieceHeader, 32> BasicJob::freeHeaders;

///////////////////////////////////////////////////////////////////////////////

BasicJob::BasicJob(boost::asio::io_service &service)  
    : BasicSocket(service)
    , onComplete( NULL)
    , onError(NULL)
{
}

///////////////////////////////////////////////////////////////////////////////

void
BasicJob::PutDataPiece( const M4D::CellBE::BasicJob::DataBuff &buf)
{
  DataBuffs bufs;
  bufs.push_back( buf);
  PutDataPiece( bufs);
}

///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////

void
BasicJob::GetDataPiece( M4D::CellBE::BasicJob::DataBuff &buf)
{
  DataBuffs bufs;
  bufs.push_back( buf);
  GetDataPiece( bufs);
}

///////////////////////////////////////////////////////////////////////////////

void
BasicJob::GetDataPiece( M4D::CellBE::BasicJob::DataBuffs &bufs)
{
  vector<boost::asio::mutable_buffer> buffers;

  // create asio buffer vector
  DataBuff *buf;
  for( DataBuffs::iterator it=bufs.begin(); it != bufs.end(); it++)
  {
    buf = (DataBuff*) & it;
    buffers.push_back( boost::asio::mutable_buffer( buf->data, buf->len) );
  }

  // read 'em
  m_socket.async_read_some(
    buffers, 
    boost::bind( & BasicJob::EndSend, this, boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////

NetStream *
BasicJob::GetNetStream( void)
{
  uint8 sourceEndian = primHeader.endian;
#ifdef LITTLE_ENDIAN
  uint8 destEndian = 0;
#else
  uint8 destEndian = 1;
#endif
  // we return NetStream instance based on source and target mashine endians
  if( sourceEndian != destEndian)
    return new UserSerializingNetStreamSwapping();
  else
    return new UserSerializingNetStreamNotSwapping();
}

///////////////////////////////////////////////////////////////////////////////