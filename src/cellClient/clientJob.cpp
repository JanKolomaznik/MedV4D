
#include <string>
#include <cstdlib>
#include <ctime>

#include "Common.h"
#include "cellBE/clientJob.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;
using namespace std;

uint32 ClientJob::lastID;

///////////////////////////////////////////////////////////////////////

ClientJob::ClientJob(
                     FilterVector &filters,
                     M4D::Imaging::AbstractDataSet *dataSet,
                     const std::string &address,
                     boost::asio::io_service &service) 
  : ClientSocket( address, service)
  , m_dataSet( dataSet)
{
  GenerateJobID();
  primHeader.action = (uint8) CREATE;
  primHeader.dataSetType = m_dataSet->_properties->GetType();

  PrimaryJobHeader::Serialize( &primHeader);

  // copy pointers to filter settings to this job
  for( FilterVector::iterator it = filters.begin();
    it != filters.end();
    it++)
  {
    m_filters.push_back( *it);
  }

  // serialize dataset settings
  m_dataSet->_properties->SerializeIntoStream( m_dataSetPropsSerialized);
  secHeader.dataSetPropertiesLen = (uint16) m_dataSetPropsSerialized.size();

  SendHeaders();

  // serialize dataSet using this job
  m_dataSet->Serialize( this);
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
ClientJob::SendHeaders( void)
{
  // prepare serialization of filters & settings
  SerializeFiltersSetting();

  SecondaryJobHeader::Serialize( &secHeader);

  // create vector of serialized information to pass to sigle send operation
  vector<boost::asio::const_buffer> buffers;
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&primHeader, sizeof(PrimaryJobHeader)) );
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&secHeader, sizeof(SecondaryJobHeader)) );
  buffers.push_back( 
    boost::asio::buffer( 
      &filterSettingsSerialized[0], filterSettingsSerialized.size() ));

  // send the buffer vector
  m_socket.async_write_some( 
    buffers, 
    boost::bind( &ClientJob::EndSend, this,
      boost::asio::placeholders::error)
  );
}

///////////////////////////////////////////////////////////////////////

void
ClientJob::SerializeFiltersSetting( void)
{
  if( m_filters.empty() )
  {
    LOG( "Empty filter vector. Add some filter into job!");
    throw ExceptionBase( "Empty filter vector. Add some filter into job!");
  }

  for( FilterVector::iterator it = m_filters.begin();
    it != m_filters.end(); it++)
  {
    (*it)->Serialize( filterSettingsSerialized);
  }

  secHeader.filterSettStreamLen = (uint16) filterSettingsSerialized.size();
}

///////////////////////////////////////////////////////////////////////