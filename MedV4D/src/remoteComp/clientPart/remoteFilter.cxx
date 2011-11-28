/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file RemoteFilter.tcc 
 * @{ 
 **/

#include "Imaging/DataSetFactory.h"

#ifndef _REMOTE_FILTER_H
#error File remoteFilter.cxx cannot be included directly!
#else

namespace M4D
{
namespace RemoteComputing
{

template< typename InputImageType, typename OutputImageType >
RemoteFilter< InputImageType, OutputImageType >
::RemoteFilter(iRemoteFilterProperties *properties)
  : PredecessorType( new typename PredecessorType::Properties() )
  , properties_(properties)
  , m_socket_(m_io_service)
  , netAccessor_(m_socket_)
{
	this->_name = "Remote Filter";
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
RemoteFilter< InputImageType, OutputImageType >
::~RemoteFilter()
{
	m_socket_.close();
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
RemoteFilter< InputImageType, OutputImageType >::PrepareOutputDatasets(void)
{
	PredecessorType::PrepareOutputDatasets();
	
	// copy properties of input image to output image
	{
		int32 minimums[OutputImageType::Dimension];
		int32 maximums[OutputImageType::Dimension];
		float32 voxelExtents[OutputImageType::Dimension];
	
		for( unsigned i=0; i < OutputImageType::Dimension; ++i ) 
		{
			minimums[i] = this->in->GetDimensionExtents(i).minimum;
			maximums[i] = this->in->GetDimensionExtents(i).maximum;
			voxelExtents[i] = this->in->GetDimensionExtents(i).elementExtent;
		}
		this->SetOutputImageSize( minimums, maximums, voxelExtents );
	}
	
	try {
		// discart old connection
		Disconnect();
		
		// create new
		Connect();
	
		SendCommand(CREATE);
	
		IO::OutStream stream(&netAccessor_);
		properties_->SerializeClassInfo(stream);
		
		SendCommand(DATASET);
		SendDataSet();
			
	} catch (asio::system_error &e) {		
		if(e.code() == asio::error::eof )
		{
			LOG("Server disconnected ...");
		}
		else
		{
			LOG("ASIO system exception, code" << e.code());
		}
	}	
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
bool
RemoteFilter< InputImageType, OutputImageType >
::ProcessImage(
		const InputImageType 	&in,
		OutputImageType		&out
		)
{
	try {
		SendCommand(EXEC);
		IO::OutStream stream(&netAccessor_);
		properties_->SerializeProperties(stream);
		return RecieveDataSet();
	} catch (asio::system_error &e) {		
		if(e.code() == asio::error::eof )
		{
			LOG("Server disconnected ...");
		}
		else
		{
			LOG("ASIO system exception, code" << e.code());
		}
		return false;
	}
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
RemoteFilter< InputImageType, OutputImageType >::Connect(void)
{
  asio::ip::tcp::resolver resolver(m_io_service);

  std::stringstream port;
  port << SERVER_PORT;

  asio::ip::tcp::resolver::query query(
	FindAvailableServer(),
    port.str(),
    asio::ip::tcp::resolver::query::numeric_service);

  asio::ip::tcp::resolver::iterator endpoint_iterator = 
	  resolver.resolve(query);
  asio::ip::tcp::resolver::iterator end;

  asio::error_code error = asio::error::host_not_found;
  while (error && endpoint_iterator != end)
  {
	m_socket_.close();
	m_socket_.connect(*endpoint_iterator++, error);
  }
  if (error)
    throw M4D::ErrorHandling::ExceptionBase(
      "Not able to connect to Cell sever");
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputImageType, typename OutputImageType >
void
RemoteFilter< InputImageType, OutputImageType >::Disconnect()
{
	if(m_socket_.is_open())
		m_socket_.close();
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
RemoteFilter< InputImageType, OutputImageType >::SendDataSet(void)
{
	try {
		IO::OutStream stream(&netAccessor_);
		M4D::Imaging::DataSetFactory::SerializeDataset(stream, *this->in);
	} catch (asio::system_error &e) {		
		if(e.code() == asio::error::eof )
		{
			LOG("Server disconnected ...");
		}
		else
		{
			LOG("ASIO system exception, code" << e.code());
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
bool
RemoteFilter< InputImageType, OutputImageType >::RecieveDataSet(void)
{
	IO::InStream stream(&netAccessor_);
	
	uint8 result;
	stream.Get<uint8>(result);
	
	switch ( (eRemoteComputationResult) result) 
	{
		case OK:
			M4D::Imaging::DataSetFactory::DeserializeDataset(stream, *this->out);
			return true;
			break;
			
		case FAILED:
			return false;
			break;
		default:
			ASSERT(false);
	}
	return false;	// toremove WARNING about no val returning
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
RemoteFilter< InputImageType, OutputImageType >::SendCommand( eCommand command)
{
	try {
		m_socket_.write_some( 
				asio::buffer( (uint8*)&command, sizeof(uint8)) );
	} catch (asio::system_error &e) {		
		if(e.code() == asio::error::eof )
		{
			LOG("Server disconnected ...");
		}
		else
		{
			LOG("ASIO system exception, code" << e.code());
		}
	}
}
///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_REMOTE_FILTER_H*/

/** @} */

