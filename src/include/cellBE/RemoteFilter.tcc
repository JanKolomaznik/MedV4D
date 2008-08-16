#ifndef _REMOTE_FILTER_H
#error File RemoteFilter.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

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
  // create dataSetSerializers for input & output dataSets if the not already..
  if( m_inSerializer == NULL)
    m_inSerializer = GeneralDataSetSerializer::GetDataSetSerializer( 
      (M4D::Imaging::AbstractDataSet *) &in);
  if( m_outSerializer == NULL)
    m_outSerializer = GeneralDataSetSerializer::GetDataSetSerializer( &out);

  m_inSerializer->SetDataSet( (M4D::Imaging::AbstractDataSet *) &in);
  m_outSerializer->SetDataSet( &out);

  m_job->SendDataSet();
  m_job->SendExecute();

	return true;
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_REMOTE_FILTER_H*/
