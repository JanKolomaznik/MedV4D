
#include "Common.h"
#include "serverJob.h"

#include "Imaging/dataSetProperties.h"
#include "Imaging/ImageFactory.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::DeserializeFilterSettings( void)
{
  uint8 filterID;
  AbstractFilterSetting *fs;

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
        m_filters.push_back( fs);
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

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::ReadSecondaryHeader()
{
  m_socket.async_read_some(
      boost::asio::buffer( (uint8*)&secHeader, sizeof( SecondaryJobHeader) ),
      boost::bind( &ServerJob::EndSecondaryHeaderRead, this, 
        boost::asio::placeholders::error)
      );
}

///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::EndJobSettingsRead( const boost::system::error_code& error)
{
  try {
    HandleErrors( error);
    DeserializeFilterSettings();

    // build the pipeline according filterSettingsVector
    BuildThePipeLine();
    // create appropriate dataSet according secHeader.dataSetType
    CreateDataSet();

    // read the dataSet properties
    m_socket.async_read_some(
      boost::asio::buffer( (void*) &dataSet->_properties, secHeader.dataSetPropertiesLen),
      boost::bind( &ServerJob::EndDataSetPropertiesRead, this,
        boost::asio::placeholders::error)
      );

  } catch( ExceptionBase &) {
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::EndDataSetPropertiesRead( const boost::system::error_code& error)
{
  // dataSet->DeSerialize( this);
}

///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::CreateDataSet( void)
{
  // create the dataSet instance
  switch( (DataSetType) secHeader.dataSetType)
  {
  case DATSET_IMAGE2D:
    //ImageFactory::Create
    break;

  case DATSET_IMAGE3D:
    break;

  case DATSET_IMAGE4D:
    break;
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::BuildThePipeLine( void)
{
}

///////////////////////////////////////////////////////////////////////////////
