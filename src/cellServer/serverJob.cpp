

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