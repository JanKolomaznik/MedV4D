
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
                     FilterSerializerVector &filters
                   //, AbstractDataSetSerializer *inDataSetSeralizer
                   //, AbstractDataSetSerializer *outDataSetSerializer
                   , const std::string &address
                   , boost::asio::io_service &service) 
  : ClientSocket( address, service)
  , m_inDataSetSerializer( NULL)
  , m_outDataSetSerializer( NULL)
{
  m_filters = filters;  // copy filters definitions
  SendCreate();
}

///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SendCreate( void)
{
  primHeader.action = (uint8) CREATE;
  GenerateJobID();

  SerializeFilters();
  primHeader.nexPartLength = (uint16) filterSettingsSerialized.size();

  PrimaryJobHeader::Serialize( &primHeader);

  // create vector of serialized information to pass to sigle send operation
  vector<boost::asio::const_buffer> buffers;
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&primHeader, sizeof(PrimaryJobHeader)) );
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

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SerializeFiltersProperties( void)
{
  NetStreamVector tmp;

  for( FilterSerializerVector::iterator it = m_filters.begin();
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

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SerializeFilters( void)
{
  NetStreamVector tmp;

  for( FilterSerializerVector::iterator it = m_filters.begin();
    it != m_filters.end(); it++)
  {
    filterSettingsSerialized << (uint16) (*it)->GetTypeID();  // insert filterTypeID
    filterSettingsSerialized << (*it)->GetID();
    (*it)->SerializeClassInfo( tmp); // serialize it into tmp stream

    // put size of the serialization
    filterSettingsSerialized << (uint16) tmp.size();

    // copy actual serialization into final stream from tmp
    filterSettingsSerialized.insert( 
      filterSettingsSerialized.end(), tmp.begin(), tmp.end() );
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::OnResponseRecieved( const boost::system::error_code& error
                              , ResponseHeader *header)
{
  // now we recieved response header
  try {
    HandleErrors( error);
    ResponseHeader::Deserialize( header);

    switch( (ResponseID) header->result)
    {
    case RESPONSE_OK:
      // everything was fine, so continue reading dataSetProperties
      ReadDataPeiceHeader( m_outDataSetSerializer);
      break;

    case RESPONSE_ERROR_IN_EXECUTION:
    case RESPONSE_ERROR_IN_INPUT:
      if( this->onError != NULL)  // call error handler
        onError();
      break;
    }

  } catch( ExceptionBase &) {
  }
}

///////////////////////////////////////////////////////////////////////////////

ClientJob::~ClientJob()
{
  // just send destroy request to server
  SendDestroy();
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SendDestroy( void)
{
  primHeader.action = (uint8) DESTROY;
  primHeader.nexPartLength = 0; // not used
  PrimaryJobHeader::Serialize( &primHeader);

  m_socket.async_write_some(
    boost::asio::buffer( (uint8*) &primHeader, sizeof( PrimaryJobHeader) )
    , boost::bind( & ClientJob::EndSend, this, boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SendExecute( void)
{
  //primHeader.action = (uint8) EXEC;
  //PrimaryJobHeader::Serialize( &primHeader);

  //// now everything is sent, so we have to wait for response
  //{
  //  // get free header
  //  ResponseHeader *h = m_freeResponseHeaders.GetFreeItem();

  //  // read into it
  //  m_socket.async_read_some( 
  //    boost::asio::buffer( (uint8*)h, sizeof( ResponseHeader) )
  //    , boost::bind( &ClientJob::OnResponseRecieved, this,
  //      boost::asio::placeholders::error, h)
  //  );
  //}
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SendFilterProperties( void)
{
  primHeader.action = (uint8) FILTERS;

  // prepare serialization of filters & settings
  SerializeFiltersProperties();
  primHeader.nexPartLength = (uint16) filterSettingsSerialized.size();

  PrimaryJobHeader::Serialize( &primHeader);

  // create vector of serialized information to pass to sigle send operation
  vector<boost::asio::const_buffer> buffers;
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&primHeader, sizeof(PrimaryJobHeader)) );
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

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SendDataSet( void)
{
  primHeader.action = (uint8) DATASET;

  // serialize dataset settings
  m_inDataSetSerializer->SerializeProperties( m_dataSetPropsSerialized);

  primHeader.nexPartLength = (uint16) m_dataSetPropsSerialized.size();  

  // create vector of serialized information to pass to sigle send operation
  vector<boost::asio::const_buffer> buffers;
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&primHeader, sizeof(PrimaryJobHeader)) );
  buffers.push_back( 
    boost::asio::buffer( 
      &m_dataSetPropsSerialized[0], m_dataSetPropsSerialized.size() ));

  // send the buffer vector
  m_socket.async_write_some( 
    buffers, 
    boost::bind( &ClientJob::EndSend, this,
      boost::asio::placeholders::error)
  );

  // serialize dataSet using this job
  m_inDataSetSerializer->Serialize( this);

  SendEndOfDataSetTag();
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SendPrimaryHeader( void)
{
  PrimaryJobHeader::Serialize( &primHeader);

  m_socket.async_write_some(
    boost::asio::buffer( (uint8*) &primHeader, sizeof( PrimaryJobHeader) )
    , boost::bind( & ClientJob::EndSend, this, boost::asio::placeholders::error)
    );
}

/////////////////////////////////////////////////////////////////////////////// 

void
ClientJob::SetDataSets( M4D::Imaging::AbstractDataSet *inDataSet
                  , M4D::Imaging::AbstractDataSet *outdataSet)
{
  m_inDataSet = inDataSet;
  m_outDataSet = outdataSet;

  // create dataSetSerializers for input & output dataSets if the not already..
  {
    if( m_inDataSetSerializer == NULL)
      m_inDataSetSerializer = 
        GeneralDataSetSerializer::GetDataSetSerializer( inDataSet);
    if( m_outDataSetSerializer == NULL)
      m_outDataSetSerializer = 
        GeneralDataSetSerializer::GetDataSetSerializer( outdataSet);
  }

  // assign dataSets
  m_inDataSetSerializer->SetDataSet( inDataSet);
  m_outDataSetSerializer->SetDataSet( outdataSet);

  SendDataSet();
}

///////////////////////////////////////////////////////////////////////////////