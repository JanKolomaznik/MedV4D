
#include <string>

#include "cellBE/clientJob.h"

using namespace M4D::CellBE;
using namespace std;
using boost::asio::ip::tcp;

uint32 ClientJob::lastID;

///////////////////////////////////////////////////////////////////////

ClientJob::ClientJob(
                     FilterVector &filters,
                     DataSetProperties *props,
                     const std::string &address,
                     boost::asio::io_service &service) 
  : ClientSocket( address, service)
  , onComplete( NULL)
  , onError(NULL)
{
  GenerateJobID();
  primHeader.action = (uint8) CREATE;

  // copy pointers to filter settings to this job
  for( FilterVector::iterator it = filters.begin();
    it != filters.end();
    it++)
  {
    m_filters.push_back( *it);
  }  
  
}

///////////////////////////////////////////////////////////////////////

void 
ClientJob::EndSend( const boost::system::error_code& e)
{
  try {
    HandleErrors( e);
  } catch( ExceptionBase &) {
    if( onError != NULL)
      onError();
  }
}

///////////////////////////////////////////////////////////////////////

void
ClientJob::GenerateJobID( void)
{
  NetStreamArrayBuf s( primHeader.id.id, IDLEN);

  s << ++lastID;
  //s << m_soc  // random based on host name
  //s <<        // random based on time
}

///////////////////////////////////////////////////////////////////////

void
ClientJob::SendHeaders( void)
{
  // prepare serialization of filters & settings
  SerializeFiltersSetting();

  PrimaryJobHeader::Serialize( &primHeader);
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