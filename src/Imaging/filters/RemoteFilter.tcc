#ifndef _REMOTE_FILTER_H
#error File RemoteFilter.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
RemoteFilter
::RemoteFilter()
{
	//TODO
}

template< typename InputImageType, typename OutputImageType >
RemoteFilter
::~RemoteFilter()
{

	//TODO
}

template< typename InputImageType, typename OutputImageType >
bool
RemoteFilter
::ProcessImage(
		const InputImageType 	&in,
		OutputImageType		&out
		)
{
	//TODO
	return false;
}

template< typename InputImageType, typename OutputImageType >
void
RemoteFilter
::PrepareOutputDatasets()
{
	const InputImageType 	&in = *(this->in);
	OutputImageType		&out = *(this->out);

	//TODO
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_REMOTE_FILTER_H*/
