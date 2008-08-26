#ifndef _REMOTE_FILTER_H
#error File RemoteFilter.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
RemoteFilter< InputImageType, OutputImageType >
::RemoteFilter( typename RemoteFilter< InputImageType, OutputImageType >::Properties *prop )
  : PredecessorType( prop ) 
  , m_job( NULL)
  , m_inSerializer(NULL)
  , m_outSerializer(NULL)
{
}

///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
RemoteFilter< InputImageType, OutputImageType >
::~RemoteFilter()
{
  if( m_job != NULL)
    delete m_job;
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
  m_job->SendFilterProperties();

  // setting datSets is performed here because actual dataSet may not be created
  // before
  m_job->SetDataSets( in, out );

  // if( first)
  m_job->SendDataSet();
  // if( dataSetAlreadySent)
  //m_job->SendExecute();

	return true;
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_REMOTE_FILTER_H*/
