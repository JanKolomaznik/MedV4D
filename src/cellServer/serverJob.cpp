
#include "Common.h"
#include "serverJob.h"
#include "jobManager.h"

#include "Imaging/ImageFactory.h"
#include <vector>


using namespace M4D::CellBE;
using namespace M4D::Imaging;
using namespace std;

///////////////////////////////////////////////////////////////////////////////

ServerJob::ServerJob(boost::asio::ip::tcp::socket *sock
                     , JobManager* jobManager)
    : BasicJob(sock)
    , m_jobManager( jobManager)
{
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::EndReadPipelineDefinition( const boost::system::error_code& error)
{
  try {
    HandleErrors( error);
    
    NetStreamArrayBuf s( 
      &m_filterSettingContent[0], m_filterSettingContent.size());

    AbstractFilterSerializer *fSeriz;
    // create filter instances according filter properties in stream
    // and add them into pipeline
    AbstractPipeFilter *producer, *consumer;
    if( s.HasNext() )   // add the first
    {
      // perform deserialization
      GeneralFilterSerializer::DeSerialize( &producer, &fSeriz, s);
      m_pipeLine.AddFilter( producer);

      // add created Serializer into Map
      m_filterSeralizersMap.insert( FilterSerializersMap::value_type(
        fSeriz->GetID(), fSeriz) );

      m_pipelineBegin = producer;
      consumer = producer;
    }

    // now for each remaining create & connect with predecessing
    while(s.HasNext())
    {
      // perform deserialization
      GeneralFilterSerializer::DeSerialize( &consumer, &fSeriz, s);
      // add it into PipeLine
      m_pipeLine.AddFilter( consumer);
      m_pipeLine.MakeConnection( *producer, 0, *consumer, 0);

      // add created Serializer into Map
      m_filterSeralizersMap.insert( FilterSerializersMap::value_type(
        fSeriz->GetID(), fSeriz) );
    }

    m_pipelineEnd = consumer;

    m_state = IDLE;
    SendResultBack( RESPONSE_OK, m_state);

    WaitForCommand();

  } catch( WrongFilterException &) {
    m_state = CREATION_FAILED;
    SendResultBack( RESPONSE_FAILED, m_state);
  } catch( ExceptionBase &) {
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::DeserializeFilterProperties( void)
{
  NetStreamArrayBuf s( &m_filterSettingContent[0], 
    m_filterSettingContent.size());

  uint16 id;
  FilterSerializersMap::iterator found;

  if( s.HasNext() )   // add the first
  {
    s >> id;
    found = m_filterSeralizersMap.find( id);
    if( found == m_filterSeralizersMap.end() )
      throw WrongFilterException();
    
    found->second->DeSerializeProperties( s);
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::ReadPipelineDefinition( void)
{
  m_filterSettingContent.resize( primHeader.nexPartLength);
  m_socket->async_read_some(
    boost::asio::buffer( m_filterSettingContent),
    boost::bind( &ServerJob::EndReadPipelineDefinition, this,
      boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::ReadFilters( void)
{
  m_filterSettingContent.resize( primHeader.nexPartLength);
  m_socket->async_read_some(
    boost::asio::buffer( m_filterSettingContent),
    boost::bind( &ServerJob::EndFiltersRead, this,
      boost::asio::placeholders::error)
    );
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::ReadDataSet( void)
{
  // read the dataSet properties
  m_filterSettingContent.resize( primHeader.nexPartLength);
  m_socket->async_read_some(
    boost::asio::buffer( m_filterSettingContent),
    boost::bind( &ServerJob::EndDataSetPropertiesRead, this,
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

    m_state = FILTER_PROPS_OK;
    SendResultBack( RESPONSE_OK, m_state);

    WaitForCommand();

  } catch( WrongFilterException &) {
    m_state = FILTER_PROPS_WRONG;
    SendResultBack( RESPONSE_FAILED, m_state);
  } catch( ExceptionBase &) {
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::EndDataSetPropertiesRead( const boost::system::error_code& error)
{
  try {
    HandleErrors( error);

    NetStreamArrayBuf s( &m_filterSettingContent[0], 
    m_filterSettingContent.size());

    // create the dataSet
    // TODO destroy old dataSEt ..
    AbstractDataSetSerializer *dsSerializer = NULL;
    GeneralDataSetSerializer::DeSerializeDataSetProperties( 
      &dsSerializer, &m_inDataSet, s);

    // connect it to pipeline
    m_pipeLine.MakeInputConnection( 
      *m_pipelineBegin, 0, AbstractDataSet::ADataSetPtr( m_inDataSet) );

    // create and connect output dataSet
    ConnectionInterface &conn = 
      m_pipeLine.MakeOutputConnection( *m_pipelineEnd, 0, true);
    m_outDataSet = &conn.GetDataset();
    // add message listener to be able catch execution done or failed messages
    conn.SetMessageHook( 
      MessageReceiverInterface::Ptr( new ExecutionDoneCallback(this) ) );

    // lock the whole dataSet (no progression yet so whole dataSet)
    WriterBBoxInterface &lock = ((AbstractImage *)m_inDataSet)->SetWholeDirtyBBox();

    // and execute the pipeline. Actual exectution will wait to whole
    // dataSet unlock when whole dataSet is read (don't forget to do it!!)
    m_pipelineBegin->Execute();

    m_state = DATASET_OK;
    SendResultBack( RESPONSE_OK, m_state);

    // now start recieving actual data using the retrieved serializer
    ReadDataPeiceHeader( dsSerializer);
  
  } catch( NetException &) {
  } catch( WrongDSetException &) {
    m_state = DATASET_WRONG;
    SendResultBack( RESPONSE_FAILED, m_state);
  } catch( ExceptionBase &) {
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::SendResultBack( ResponseID result, State state)
{
  vector<boost::asio::const_buffer> buffers; 

  ResponseHeader *h = m_freeResponseHeaders.GetFreeItem();
  buffers.push_back( 
      boost::asio::buffer( (uint8*)h, sizeof(ResponseHeader)) );

  h->result = (uint8) result;

  AbstractDataSetSerializer *outSerializer = NULL;

  switch( result)
  {
  case RESPONSE_DATASET:
    // get dataSetSerializer ...
    outSerializer = 
      GeneralDataSetSerializer::GetDataSetSerializer( m_outDataSet);

    // serialize dataset settings
    outSerializer->SerializeProperties( m_dataSetPropsSerialized);

    h->resultPropertiesLen = (uint16) m_dataSetPropsSerialized.size();
    break;

  default:
    h->resultPropertiesLen = (uint16) state;
    break;
  }

  ResponseHeader::Serialize( h);

  // send the buffer vector
  m_socket->async_write_some( 
    buffers, 
    boost::bind( &ServerJob::OnResultHeaderSent, this,
      boost::asio::placeholders::error, h)
      );

  // start sending dataSet if RESPONSE_EXEC_COMPLETE
  if( result == RESPONSE_DATASET)
  {    
    outSerializer->Serialize( this);
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::OnResultHeaderSent( const boost::system::error_code& error
    , ResponseHeader *h)
{
  try {
    HandleErrors( error);

    m_freeResponseHeaders.PutFreeItem( h);  // return the header to pool

  } catch( NetException &) {
    m_freeResponseHeaders.PutFreeItem( h);
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::OnExecutionDone( void)
{
  SendResultBack( RESPONSE_DATASET, EXECUTED);
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::OnExecutionFailed( void)
{
  m_state = FAILED;
  SendResultBack( RESPONSE_FAILED, m_state);
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::Command( PrimaryJobHeader *header)
{
  switch( header->action)
  {
  case BasicJob::DATASET:
    LOG( "DATASET reqest arrived");
    ReadDataSet();
    break;

  case BasicJob::FILTERS:
    LOG( "FILTERS reqest arrived");
    ReadFilters();    
    break;

  case BasicJob::ABORT:
    LOG( "ABORT reqest arrived");
    AbortComputation();    
    break;

  //case BasicJob::EXEC:
  //  Execute();
  //  LOG( "EXEC reqest arrived");
  //  break;

  case BasicJob::DESTROY:
      LOG( "DESTROY reqest arrived");
      m_jobManager->RemoveJob( header->id );
      break;

  default:
    LOG( "Unrecognized action job action."); // From: " << m_socket );
    throw ExceptionBase("Unrecognized action job action");
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::AbortComputation( void)
{
  m_pipeLine.StopFilters();
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::WaitForCommand( void)
{  
  m_socket->async_read_some(
      boost::asio::buffer( (uint8*) &primHeader, sizeof(PrimaryJobHeader) ),
      boost::bind( &ServerJob::EndWaitForCommand, this, boost::asio::placeholders::error)
      );
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::EndWaitForCommand( const boost::system::error_code& error)
{
  try {
    HandleErrors( error);

    Command( &primHeader);

  } catch( NetException &) {
  }
}

///////////////////////////////////////////////////////////////////////////////