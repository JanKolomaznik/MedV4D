
#include <string>
#include <cstdlib>
#include <ctime>

#include "Common.h"
#include "cellBE/clientJob.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;
using namespace std;

uint32 ClientJob::lastID;

///////////////////////////////////////////////////////////////////////////////

ClientJob::ClientJob(
                     FilterPropsVector &filters,
                     M4D::Imaging::AbstractDataSet *dataSet,
                     const std::string &address,
                     boost::asio::io_service &service) 
  : ClientSocket( address, service)
  , m_dataSet( dataSet) // copy dataSet pointer
{
  m_filters = filters;  // copy filters definitions
  SendCreate();
}

///////////////////////////////////////////////////////////////////////

void
ClientJob::GenerateJobID( void)
{
  NetStreamArrayBuf s( primHeader.id.id, IDLEN);

  s << ++lastID;

  // random based on host name
  boost::asio::ip::address_v4::bytes_type bytes;
  bytes = m_socket.local_endpoint().address().to_v4().to_bytes();
  uint32 seed = bytes[0];
  for( int i=1; i<4; i++)
  {
    seed <<= 8;
    seed += bytes[i];
  }

  srand( seed);
  s << (uint32) rand();  
  
  // random based on time
  srand( (uint32) time(0));  
  s << (uint32) rand();
}

///////////////////////////////////////////////////////////////////////

void
ClientJob::SendCreate( void)
{
  primHeader.action = (uint8) CREATE;  
  GenerateJobID();

  // prepare serialization of filters & settings
  SerializeFiltersProperties();
  primHeader.filterSettStreamLen = (uint16) filterSettingsSerialized.size();

  // serialize dataset settings
  GeneralDataSetSerializer::SerializeDataSetProperties( 
    m_dataSet, 
    m_dataSetPropsSerialized);

  primHeader.dataSetPropertiesLen = (uint16) m_dataSetPropsSerialized.size();

  PrimaryJobHeader::Serialize( &primHeader);

  // create vector of serialized information to pass to sigle send operation
  vector<boost::asio::const_buffer> buffers;
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&primHeader, sizeof(PrimaryJobHeader)) );
  buffers.push_back( 
    boost::asio::buffer( 
      &m_dataSetPropsSerialized[0], m_dataSetPropsSerialized.size() ));
  buffers.push_back( 
    boost::asio::buffer( 
      &filterSettingsSerialized[0], filterSettingsSerialized.size() ));

  // send the buffer vector
  m_socket.async_write_some( 
    buffers, 
    boost::bind( &ClientJob::EndSend, this,
      boost::asio::placeholders::error)
  );

  // serialize dataSet using this job
  GeneralDataSetSerializer::SerializeDataSet( m_dataSet, this);

  // send EndingTag telling no more data will come
  m_socket.async_write_some( 
    boost::asio::buffer(
      (uint8*)&endHeader, sizeof( DataPieceHeader) ),
    boost::bind( &ClientJob::EndSend, this,
      boost::asio::placeholders::error)
  );
}

///////////////////////////////////////////////////////////////////////

void
ClientJob::SerializeFiltersProperties( void)
{
  if( m_filters.empty() )
  {
    LOG( "Empty filter vector. Add some filter into job!");
    throw ExceptionBase( "Empty filter vector. Add some filter into job!");
  }

  NetStreamVector tmp;

  for( FilterPropsVector::iterator it = m_filters.begin();
    it != m_filters.end(); it++)
  {
    filterSettingsSerialized << (uint16) (*it)->GetID();  // insert filterID
    (*it)->SerializeProperties( tmp); // serialize it into tmp stream

    // put size of the serialization
    filterSettingsSerialized << (uint16) tmp.size();

    // copy actual serialization into final stream from tmp
    filterSettingsSerialized.insert( 
      filterSettingsSerialized.end(), tmp.begin(), tmp.end() );
  }
}

///////////////////////////////////////////////////////////////////////