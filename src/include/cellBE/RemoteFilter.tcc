#ifndef _REMOTE_FILTER_H
#error File RemoteFilter.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
RemoteFilter
::~RemoteFilter()
{
  if( m_job != NULL)
    delete m_job;
}

template< typename InputImageType, typename OutputImageType >
bool
RemoteFilter
::ProcessImage(
		const InputImageType 	&in,
		OutputImageType		&out
		)
{
  m_job->SendExecute();
	return true;
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_REMOTE_FILTER_H*/
