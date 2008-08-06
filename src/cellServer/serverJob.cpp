
#include "Common.h"
#include "serverJob.h"

#include "Imaging/ImageFactory.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::DeserializeFilterProperties( void)
{
  NetStreamArrayBuf s( &m_filterSettingContent[0], 
    m_filterSettingContent.size());

  // create filter instances according filter properties instream
  AbstractFilter *filter;
  try {
    while(s.HasNext())
    {
      filter = GeneralFilterSerializer::DeSerialize( s);  // TODO
    }
  } catch( ExceptionBase) {
    // do nothing. Just continue
    return;
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::ReadFilters( void)
{
  m_filterSettingContent.resize( primHeader.filterSettStreamLen);
  m_socket.async_read_some(
    boost::asio::buffer( m_filterSettingContent),
    boost::bind( &ServerJob::EndFiltersRead, this,
      boost::asio::placeholders::error)
    );
}



///////////////////////////////////////////////////////////////////////////////

void
ServerJob::EndFiltersRead( const boost::system::error_code& error)
{
  try {
    HandleErrors( error);
    DeserializeFilterProperties();

    // build the pipeline according filterSettingsVector
    BuildThePipeLine();

    // read the dataSet properties
    m_filterSettingContent.resize( primHeader.dataSetPropertiesLen);
    m_socket.async_read_some(
      boost::asio::buffer( m_filterSettingContent),
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
  // create appropriate dataSet according
  CreateDataSet();
}


///////////////////////////////////////////////////////////////////////////////

void
ServerJob::CreateDataSet( void)
{
  NetStreamArrayBuf s( &m_filterSettingContent[0], 
    m_filterSettingContent.size());

  GeneralDataSetSerializer::DeSerializeDataSetProperties(
    m_dataSet, s);

  // now start recieving actual data
  
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::BuildThePipeLine( void)
{
}

///////////////////////////////////////////////////////////////////////////////
