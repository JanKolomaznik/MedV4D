

#include "serverJob.h"

using namespace M4D::CellBE;

///////////////////////////////////////////////////////////////////////

void
ServerJob::DeserializeFilterSettings( void)
{
  uint8 filterID;
  FilterSetting *fs;

  NetStreamArrayBuf s( &m_filterSettingContent[0], 
    m_filterSettingContent.size());

  try {
    while(1)  // it's breaked by exception when reading behind stream
    {
      s >> filterID;
      switch( (FilterID) filterID)
      {
      case Thresholding:
        fs = new ThresholdingSetting();
        fs->DeSerialize(s);
        filters.push_back( fs);
        break;

      default:
        LOG( "Unrecognized filter");
        throw ExceptionBase("Unrecognized filter");
      }
    }
  } catch( ExceptionBase) {
    // do nothing. Just continue
    return;
  }
}

///////////////////////////////////////////////////////////////////////

void
ServerJob::ReadSecondaryHeader()
{
  m_socket.async_read_some(
      boost::asio::buffer( (uint8*)&secHeader, sizeof( SecondaryJobHeader) ),
      boost::bind( &ServerJob::EndSecondaryHeaderRead, this, 
        boost::asio::placeholders::error)
      );
}

///////////////////////////////////////////////////////////////////////

void
ServerJob::ReadDataPeiceHeader( void)
{
  DataPieceHeader *header = freeHeaders.GetFreeItem();

  m_socket.async_read_some(
      boost::asio::buffer( (uint8*)header, sizeof( DataPieceHeader) ),
      boost::bind( &ServerJob::EndReadDataPeiceHeader, this, 
        boost::asio::placeholders::error, header)
      );
}

///////////////////////////////////////////////////////////////////////

void
ServerJob::EndSecondaryHeaderRead( const boost::system::error_code& error)
{
  try {
    HandleErrors( error);

    SecondaryJobHeader::Deserialize(&secHeader);

    m_filterSettingContent.resize( secHeader.filterSettStreamLen);
    m_socket.async_read_some(
      boost::asio::buffer( m_filterSettingContent),
      boost::bind( &ServerJob::EndJobSettingsRead, this,
        boost::asio::placeholders::error)
      );
  } catch( ExceptionBase &) {
  }

}

///////////////////////////////////////////////////////////////////////

void
ServerJob::EndJobSettingsRead( const boost::system::error_code& error)
{
  try {
    HandleErrors( error);
    DeserializeFilterSettings();

  } catch( ExceptionBase &) {
  }
}

///////////////////////////////////////////////////////////////////////

void
ServerJob::EndReadDataPeiceHeader( const boost::system::error_code& error,
                                  DataPieceHeader *header)
{
  try {
    HandleErrors( error);
    DataPieceHeader::Deserialize( header);

  } catch( ExceptionBase &) {
  }
}

///////////////////////////////////////////////////////////////////////

//void
//ClientJob::PutDataPiece( const M4D::CellBE::BasicJob::DataBuff &buf)
//{
//  DataBuffs bufs;
//  bufs.push_back( buf);
//  PutDataPiece( bufs);
//}
//
/////////////////////////////////////////////////////////////////////////
//
//void
//ClientJob::PutDataPiece( const M4D::CellBE::BasicJob::DataBuffs &bufs)
//{
//  // count total length of all buffers
//  size_t totalLen = 0;
//
//  DataPieceHeader *header = p_freeDPHeaders.back();
//  p_freeDPHeaders.pop_back();
//
//  vector<boost::asio::const_buffer> buffers;
//
//  // push header of all dataPiece
//  buffers.push_back( boost::asio::buffer( header, sizeof( DataPieceHeader)) );
//
//  // push rest of bufs
//  const DataBuff *buf;
//  for( DataBuffs::const_iterator it=bufs.begin(); it != bufs.end(); it++)
//  {
//    buf = (DataBuff*) & it;
//    buffers.push_back( boost::asio::const_buffer( buf->data, buf->len) );
//    totalLen += buf->len;
//  }
//
//  header->pieceSize = (uint32) totalLen;
//  DataPieceHeader::Serialize( header);
//
//  // get free dataPieceHeader
//  m_socket.async_write_some(
//    buffers, 
//    boost::bind( & ClientJob::EndSend, this, boost::asio::placeholders::error)
//    );
//}

///////////////////////////////////////////////////////////////////////