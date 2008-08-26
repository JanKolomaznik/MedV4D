
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

      // set starting by message
      consumer->SetUpdateInvocationStyle( 
        AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );

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

  } catch( NetException &ne) {
    LOG( "NetException in EndReadPipelineDefinition" << ne.what() );
  } catch( WrongFilterException &wfe) {
    m_state = CREATION_FAILED;
    SendResultBack( RESPONSE_FAILED, m_state);
    LOG( "WrongFilterException in EndReadPipelineDefinition" << wfe.what() );
  } catch( ExceptionBase &e) {
    LOG( "ExceptionBase in EndReadPipelineDefinition" << e.what() );
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

  while( s.HasNext() )   // add the first
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

  } catch( NetException &ne) {
    LOG( "NetException in EndFiltersRead" << ne.what() );
  } catch( WrongFilterException &wfe) {
    m_state = FILTER_PROPS_WRONG;
    SendResultBack( RESPONSE_FAILED, m_state);
    LOG( "WrongFilterException in EndFiltersRead" << wfe.what() );
  } catch( ExceptionBase &e) {
    LOG( "ExceptionBase in EndFiltersRead" << e.what() );
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
    AbstractDataSet::ADataSetPtr inputDataSet;
    GeneralDataSetSerializer::DeSerializeDataSetProperties( 
      &m_inDataSetSerializer, &inputDataSet, s);

    // connect it to pipeline
    m_pipeLine.MakeInputConnection( *m_pipelineBegin, 0, inputDataSet);

    // create and connect created output dataSet
    ConnectionInterface &conn = 
      m_pipeLine.MakeOutputConnection( *m_pipelineEnd, 0, true);
    // create outDataSerializer
    m_outDataSetSerializer = GeneralDataSetSerializer::GetDataSetSerializer(
      &conn.GetDataset() );
    // add message listener to be able catch execution done or failed messages
    conn.SetMessageHook( 
      MessageReceiverInterface::Ptr( new ExecutionDoneCallback(this) ) );

    // lock the whole dataSet (no progression yet so whole dataSet)
    // NOte AbstractImage is used because no universal locking is used
    // and current implementaton has only images
    AbstractImage *imagePointer = (AbstractImage *) inputDataSet.get();
    D_PRINT("Locking DS");
    m_DSLock = &imagePointer->SetWholeDirtyBBox();

    // and execute the pipeline. Actual exectution will wait to whole
    // dataSet unlock when whole dataSet is read (don't forget to do it!!)
    m_pipelineBegin->Execute();

    m_state = DATASET_PROPS_OK;
    SendResultBack( RESPONSE_OK, m_state);

    // now start recieving actual data using the retrieved serializer
    ReadDataPeiceHeader( m_inDataSetSerializer);
  
  } catch( NetException &ne) {
    LOG( "NetException in EndDataSetPropertiesRead" << ne.what() );
  } catch( WrongDSetException &wdse) {
    m_state = DATASET_PROPS_WRONG;
    SendResultBack( RESPONSE_FAILED, m_state);
    LOG( "WrongDSetException in EndDataSetPropertiesRead" << wdse.what() );
  } catch( ExceptionBase &e) {
    LOG( "ExceptionBase in EndDataSetPropertiesRead" << e.what() );
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::SendResultBack( ResponseID result, State state)
{
  ResponseHeader *h = m_freeResponseHeaders.GetFreeItem();

  h->result = (uint8) result;
  h->resultPropertiesLen = (uint16) state;

  ResponseHeader::Serialize( h);

  // send the buffer vector
  m_socket->async_write_some( 
    boost::asio::buffer( (uint8 *) h, sizeof(ResponseHeader) ), 
    boost::bind( &ServerJob::OnResultHeaderSent, this,
      boost::asio::placeholders::error, h)
      );
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::OnResultHeaderSent( const boost::system::error_code& error
    , ResponseHeader *h)
{
  try {
    HandleErrors( error);

    m_freeResponseHeaders.PutFreeItem( h);  // return the header to pool

  } catch( NetException &ne) {
    m_freeResponseHeaders.PutFreeItem( h);
    LOG( "NetException in OnResultHeaderSent" << ne.what() );
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::OnExecutionDone( void)
{
  // dump dataSet
  D_PRINT("Dumping incoming dataSet:" << endl << endl);
  D_COMMAND( m_inDataSetSerializer->Dump() );

  // dump dataSet
  D_PRINT("Dumping outcoming dataSet:" << std::endl << endl);
  D_COMMAND( m_outDataSetSerializer->Dump() );

  SendResultBack( RESPONSE_OK, EXECUTED);

  // start sending back resulting dataSet
  m_outDataSetSerializer->Serialize( this);

  SendEndOfDataSetTag();

  m_state = IDLE;
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

  case BasicJob::EXEC:
    LOG( "EXEC reqest arrived");
    Execute();    
    break;

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
ServerJob::Execute( void)
{
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

    // parse primary job header
    PrimaryJobHeader::Deserialize( &primHeader);

    Command( &primHeader);

  } catch( NetException &ne) {
    LOG( "NetException in EndWaitForCommand" << ne.what() );
  }
}

///////////////////////////////////////////////////////////////////////////////

void
ServerJob::OnDSRecieved( void)
{  
  m_state = DATASET_OK;
  SendResultBack( RESPONSE_OK, m_state);

  D_PRINT("UNLocking DS");
  m_DSLock->SetModified();     // unlock locked dataSet to start execution
}

///////////////////////////////////////////////////////////////////////////////