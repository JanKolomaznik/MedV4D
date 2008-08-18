
#include "Common.h"
#include "serverJob.h"

#include "Imaging/ImageFactory.h"
#include <vector>

using namespace M4D::CellBE;
using namespace M4D::Imaging;
using namespace std;

///////////////////////////////////////////////////////////////////////////////

ServerJob::ServerJob(boost::asio::io_service &service)
    : BasicJob(service)
{
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::DeserializeFilterPropertiesAndBuildPipeline( void)
{
  NetStreamArrayBuf s( &m_filterSettingContent[0], 
    m_filterSettingContent.size());

  // create filter instances according filter properties in stream
  // and add them into pipeline
  AbstractPipeFilter *producer, *consumer;
  if( s.HasNext() )   // add the first
  {
    producer = GeneralFilterSerializer::DeSerialize( s);
    m_pipelineBegin = producer;
    m_pipeLine.AddFilter( producer);
  }

  // now for each remaining create & connect with predecessing
  while(s.HasNext())
  {
    consumer = GeneralFilterSerializer::DeSerialize( s);
    m_pipeLine.AddFilter( consumer);
    m_pipeLine.MakeConnection( *producer, 0, *consumer, 0);
  }

  m_pipelineEnd = consumer;
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::ReadFilters( void)
{
  m_filterSettingContent.resize( primHeader.nexPartLength);
  m_socket.async_read_some(
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
  m_socket.async_read_some(
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
    DeserializeFilterPropertiesAndBuildPipeline();

  } catch( WrongFilterException &) {
    SendResultBack( RESPONSE_ERROR_IN_INPUT);
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

    // now start recieving actual data using the retrieved serializer
    ReadDataPeiceHeader( dsSerializer);
  
  } catch( NetException &) {
  } catch( WrongDSetException &) {
    SendResultBack( RESPONSE_ERROR_IN_INPUT);
  } catch( ExceptionBase &) {
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::SendResultBack( ResponseID result)
{
  ResponseHeader *h = m_freeResponseHeaders.GetFreeItem();
  h->result = (uint8) result;

  vector<boost::asio::const_buffer> buffers;
  buffers.push_back( 
      boost::asio::buffer( (uint8*)h, sizeof(ResponseHeader)) );

  AbstractDataSetSerializer *outSerializer = NULL;

  switch( result)
  {
  case RESPONSE_OK:
    // get dataSetSerializer ...
    outSerializer = 
      GeneralDataSetSerializer::GetDataSetSerializer( m_outDataSet);

    // serialize dataset settings
    outSerializer->SerializeProperties( m_dataSetPropsSerialized);

    h->resultPropertiesLen = (uint16) m_dataSetPropsSerialized.size();
    ResponseHeader::Serialize( h);
    
    buffers.push_back( 
      boost::asio::buffer( 
        &m_dataSetPropsSerialized[0], m_dataSetPropsSerialized.size() ));    
    break;

  case RESPONSE_ERROR_IN_EXECUTION:    
  case RESPONSE_ERROR_IN_INPUT:
    ResponseHeader::Serialize( h);
    break;
  }

  // send the buffer vector
  m_socket.async_write_some( 
    buffers, 
    boost::bind( &ServerJob::OnResultHeaderSent, this,
      boost::asio::placeholders::error, h)
      );

  // start sending dataSet if OK
  if( result == RESPONSE_OK)
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
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::OnExecutionDone( void)
{
  SendResultBack( RESPONSE_OK);
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::OnExecutionFailed( void)
{
  SendResultBack( RESPONSE_ERROR_IN_EXECUTION);
}

///////////////////////////////////////////////////////////////////////////////