/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file RemoteFilter.tcc 
 * @{ 
 **/

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
  : PredecessorType( &props )
  , properties_(properties)
  , m_socket_(m_io_service)
  , netAccessor_(m_socket_)
{
	Connect();
	
	SendCommand(CREATE);	
	
	Imaging::OutStream stream(&netAccessor_);
	properties_->SerializeClassInfo(stream);
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
	SendDataSet();
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
	SendCommand(EXEC);
	Imaging::OutStream stream(&netAccessor_);
	properties_->SerializeProperties(stream);
	RecieveDataSet();
	
	return true;
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
RemoteFilter< InputImageType, OutputImageType >::Connect(void)
{
  boost::asio::ip::tcp::resolver resolver(m_io_service);

  std::stringstream port;
  port << SERVER_PORT;

  boost::asio::ip::tcp::resolver::query query(
	FindAvailableServer(),
    port.str(),
    boost::asio::ip::tcp::resolver::query::numeric_service);

  boost::asio::ip::tcp::resolver::iterator endpoint_iterator = 
	  resolver.resolve(query);
  boost::asio::ip::tcp::resolver::iterator end;

  boost::system::error_code error = boost::asio::error::host_not_found;
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
RemoteFilter< InputImageType, OutputImageType >::SendDataSet(void)
{
	Imaging::OutStream stream(&netAccessor_);
	
	OutputImageType &out = GetOutputImage();
	out.SerializeProperties(stream);
	out.SerializeData(stream);
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
RemoteFilter< InputImageType, OutputImageType >::RecieveDataSet(void)
{
	Imaging::InStream stream(&netAccessor_);
	
	OutputImageType out = GetOutputImage();
	out.DeSerializeProperties(stream);
	
	//get attribs of image and call SetOutputImageSize
	{
		int32 minimums[OutputImageType::Dimension];
		int32 maximums[OutputImageType::Dimension];
		float32 voxelExtents[OutputImageType::Dimension];
	
		for( unsigned i=0; i < OutputImageType::Dimension; ++i ) 
		{
			minimums[i] = out.GetDimensionExtents(i).minimum;
			maximums[i] = out.GetDimensionExtents(i).maximum;
			voxelExtents[i] = out.GetDimensionExtents(i).elementExtent;
		}
		out.SetOutputImageSize( minimums, maximums, voxelExtents );
	}
	
	// and recieve the resulting dataset
	out.DeSerializeData(stream);
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
RemoteFilter< InputImageType, OutputImageType >::SendCommand( eCommand command)
{
	m_socket_.write_some( 
				boost::asio::buffer( (uint8*)&command, sizeof(uint8)) );
}
///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_REMOTE_FILTER_H*/

/** @} */

