
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
                   , const std::string &address
                   , boost::asio::io_service &service) 
  : ClientSocket( address, service)
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
  bytes = m_socket->local_endpoint().address().to_v4().to_bytes();
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

  SerializeRemotePipeDefinition();
  primHeader.nexPartLength = (uint16) m_remotePipeDefSerialized.size();

  PrimaryJobHeader::Serialize( &primHeader);

  // create vector of serialized information to pass to sigle send operation
  vector<boost::asio::const_buffer> buffers;
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&primHeader, sizeof(PrimaryJobHeader)) );
  buffers.push_back( 
    boost::asio::buffer( 
      &m_remotePipeDefSerialized[0], m_remotePipeDefSerialized.size() ));

  // send the buffer vector
  m_socket->async_write_some( 
    buffers, 
    boost::bind( &ClientJob::EndSend, this,
      boost::asio::placeholders::error)
  );

  ResponseHeader h;
  size_t read = m_socket->read_some(boost::asio::buffer( (uint8*) &h, sizeof(ResponseHeader)));
  ProcessResponse( h);
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SerializeFiltersProperties( void)
{
  for( FilterSerializerVector::iterator it = m_filters.begin();
    it != m_filters.end(); it++)
  {
    filterSettingsSerialized << (uint16) (*it)->GetID();  // insert filterID
    (*it)->SerializeProperties( filterSettingsSerialized); // serialize it into tmp stream
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SerializeRemotePipeDefinition( void)
{
  for( FilterSerializerVector::iterator it = m_filters.begin();
    it != m_filters.end(); it++)
  {
    m_remotePipeDefSerialized << (uint16) (*it)->GetTypeID();  // insert filterTypeID
    m_remotePipeDefSerialized << (*it)->GetID();
    (*it)->SerializeClassInfo( m_remotePipeDefSerialized);
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::ProcessResponse( const ResponseHeader &header)
{
  switch( (ResponseID) header.result)
  {
  case RESPONSE_OK:
    m_state = (State) header.resultPropertiesLen;
    break;

  case RESPONSE_FAILED:
    m_state = (State) header.resultPropertiesLen;
    if( this->onError != NULL)  // call error handler
    {
      onError();
      throw ExceptionBase("Response failed");
    }
    break;

  default:
    ASSERT( false);
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

  m_socket->write_some(
    boost::asio::buffer( (uint8*) &primHeader, sizeof( PrimaryJobHeader) )
    );
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SendFilterProperties( void)
{
  if( m_state != IDLE)
    throw WrongJobStateException();

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
  m_socket->async_write_some( 
    buffers, 
    boost::bind( &ClientJob::EndSend, this,
      boost::asio::placeholders::error)
  );

  ResponseHeader h;
  size_t read = m_socket->read_some(boost::asio::buffer( (uint8*) &h, sizeof(ResponseHeader)));
  ProcessResponse( h);
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SendDataSet( void)
{
  primHeader.action = (uint8) DATASET;

  // serialize dataset settings
  m_dataSetPropsSerialized << (uint8) m_inDataSetSerializer->GetID();
  m_inDataSetSerializer->SerializeProperties( m_dataSetPropsSerialized);

  primHeader.nexPartLength = (uint16) m_dataSetPropsSerialized.size();
  PrimaryJobHeader::Serialize( &primHeader);

  // create vector of serialized information to pass to sigle send operation
  vector<boost::asio::const_buffer> buffers;
  buffers.push_back( 
    boost::asio::buffer( (uint8*)&primHeader, sizeof(PrimaryJobHeader)) );
  buffers.push_back( 
    boost::asio::buffer( 
      &m_dataSetPropsSerialized[0], m_dataSetPropsSerialized.size() ));

  // send the buffer vector
  m_socket->async_write_some( 
    buffers, 
    boost::bind( &ClientJob::EndSend, this,
      boost::asio::placeholders::error)
  );

  ResponseHeader h;
  // read status of dataSet properties parsing
  size_t read = m_socket->read_some(boost::asio::buffer( (uint8*) &h, sizeof(ResponseHeader)));
  ProcessResponse( h);

  // send DSContent (perform ADS)
  {
    // serialize dataSet using this job
    m_inDataSetSerializer->Serialize( this);

    SendEndOfDataSetTag();
  }

  // read status of dataSet recieving
  read = m_socket->read_some(boost::asio::buffer( (uint8*) &h, sizeof(ResponseHeader)));
  ProcessResponse( h);

  D_PRINT("After DS sending: " << m_state);

  // wait for result of execution
  m_socket->read_some(boost::asio::buffer( (uint8*) &h, sizeof(ResponseHeader)));
  ProcessResponse( h);

  D_PRINT("After execution: " << m_state);

  // if execution ok, wait for result
  if( m_state == EXECUTED)
    ReadResultingDataSet();
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::ReadResultingDataSet( void)
{
  // start recieving
  DataPieceHeader h;
  m_socket->read_some(boost::asio::buffer( 
    (uint8*) &h, sizeof(DataPieceHeader)));
  DataPieceHeaderDeserialize( &h);


  DataBuffs buffs;

  while( h.pieceSize != ENDING_PECESIZE)
  {
    m_outDataSetSerializer->OnDataPieceReadRequest( &h, buffs);
    
    GetDataPiece( buffs, m_outDataSetSerializer);
    buffs.clear();

    m_socket->read_some(boost::asio::buffer( 
      (uint8*) &h, sizeof(DataPieceHeader)));
    DataPieceHeaderDeserialize( &h);
  }

  // dump dataSet
  D_PRINT("Dumping incoming dataSet:" << std::endl << endl);
  D_COMMAND (m_outDataSetSerializer->Dump() );
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::OnDSRecieved( void)
{
  m_mutex.unlock();
}

///////////////////////////////////////////////////////////////////////////////

void
ClientJob::SendExecute( void)
{
  primHeader.action = (uint8) EXEC;
  PrimaryJobHeader::Serialize( &primHeader);

  m_socket->async_write_some( 
    boost::asio::buffer( (uint8 *) &primHeader, sizeof( PrimaryJobHeader) ), 
    boost::bind( &ClientJob::EndSend, this,
      boost::asio::placeholders::error)
  );

  // read status of execution
  ResponseHeader h;
  m_socket->read_some(boost::asio::buffer( (uint8*) &h, sizeof(ResponseHeader)));
  ProcessResponse( h);

  // if execution ok, wait for result
  if( m_state == EXECUTED)
    ReadResultingDataSet();
}

/////////////////////////////////////////////////////////////////////////////// 

void
ClientJob::SetDataSets( const M4D::Imaging::AbstractDataSet &inDataSet
                  , M4D::Imaging::AbstractDataSet &outdataSet)
{
  // create dataSetSerializers for input & output dataSets if the not already..
  {
    if( m_inDataSetSerializer == NULL)
      m_inDataSetSerializer = 
      GeneralDataSetSerializer::GetDataSetSerializer( (AbstractDataSet *) &inDataSet);
    if( m_outDataSetSerializer == NULL)
      m_outDataSetSerializer = 
        GeneralDataSetSerializer::GetDataSetSerializer( &outdataSet);
  }

  // assign dataSets
  m_inDataSetSerializer->SetDataSet( (AbstractDataSet *) &inDataSet);
  m_outDataSetSerializer->SetDataSet( &outdataSet);
}

///////////////////////////////////////////////////////////////////////////////