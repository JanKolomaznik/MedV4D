
#include <vector>
#include "Common.h"
#include "cellBE/basicJob.h"

using namespace M4D::CellBE;
using namespace std;

Pool< DataPieceHeader, 32> BasicJob::freeHeaders;
Pool< ResponseHeader, 32> BasicJob::m_freeResponseHeaders;
DataPieceHeader BasicJob::endHeader(ENDING_PECESIZE);

///////////////////////////////////////////////////////////////////////////////

BasicJob::BasicJob(boost::asio::io_service &service)  
    : BasicSocket(service)
    , onComplete( NULL)
    , onError(NULL)
    , m_inDataSet( NULL)
    , m_outDataSet( NULL)
{
}

///////////////////////////////////////////////////////////////////////////////

void
BasicJob::PutDataPiece( const M4D::CellBE::DataBuff &buf)
{
  DataBuffs bufs;
  bufs.push_back( buf);
  PutDataPiece( bufs);
}

///////////////////////////////////////////////////////////////////////////////

void
BasicJob::PutDataPiece( const M4D::CellBE::DataBuffs &bufs)
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
  DataPieceHeaderSerialize( header);

  // get free dataPieceHeader
  m_socket.async_write_some(
    buffers, 
    boost::bind( & BasicJob::EndSend, this, boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////////////

void
BasicJob::GetDataPiece( M4D::CellBE::DataBuffs &bufs
                       , AbstractDataSetSerializer *dataSetSerializer)
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
    boost::bind( & BasicJob::EndReadDataPeice, this
      , boost::asio::placeholders::error
      , dataSetSerializer)
    );
}

///////////////////////////////////////////////////////////////////////////////

void
BasicJob::ReadDataPeiceHeader( AbstractDataSetSerializer *dataSetSerializer)
{
  DataPieceHeader *header = freeHeaders.GetFreeItem();

  m_socket.async_read_some(
      boost::asio::buffer( (uint8*)header, sizeof( DataPieceHeader) ),
      boost::bind( &BasicJob::EndReadDataPeiceHeader, this, 
        boost::asio::placeholders::error, header, dataSetSerializer)
      );
}

///////////////////////////////////////////////////////////////////////////////

void
BasicJob::EndReadDataPeiceHeader( const boost::system::error_code& error,
                                  DataPieceHeader *header,
                                  AbstractDataSetSerializer *dataSetSerializer)
{
  try {
    HandleErrors( error);
    DataPieceHeaderDeserialize( header);

    if( header->pieceSize == ENDING_PECESIZE)
    {
      dataSetSerializer->OnDataSetEndRead();
      // destroy serializer
      delete dataSetSerializer;

      if( this->onComplete != NULL)
        onComplete();                 // call completition callback
    }
    else
    {
      dataSetSerializer->OnDataPieceReadRequest( header, m_dataBufs);
      freeHeaders.PutFreeItem( header); // return used header back to pool
      GetDataPiece( m_dataBufs, dataSetSerializer);
    }

  } catch( ExceptionBase &) {
  }
}

///////////////////////////////////////////////////////////////////////////////
void
BasicJob::EndReadDataPeice( const boost::system::error_code& error
                           ,AbstractDataSetSerializer *dataSetSerializer)
{
  try {
    HandleErrors( error);

    ReadDataPeiceHeader( dataSetSerializer);

  } catch( ExceptionBase &) {
  }
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

void
BasicJob::SendEndOfDataSetTag( void)
{
  // send EndingTag telling no more data will come
  m_socket.async_write_some( 
    boost::asio::buffer(
      (uint8*)&endHeader, sizeof( DataPieceHeader) ),
    boost::bind( &BasicJob::EndSend, this,
      boost::asio::placeholders::error)
  );
}

///////////////////////////////////////////////////////////////////////////////