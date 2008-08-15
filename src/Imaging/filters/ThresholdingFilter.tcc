#ifndef _THRESHOLDING_FILTER_H
#error File ThresholdingFilter.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

template< typename ImageType >
ThresholdingFilter< ImageType >
::ThresholdingFilter() : PredecessorType( new Properties() )
{
	GetProperties()._functor = &(this->_elementFilter);
}

template< typename ImageType >
ThresholdingFilter< ImageType >
::ThresholdingFilter( typename ThresholdingFilter< ImageType >::Properties *prop ) : PredecessorType( prop )
{
	GetProperties()._functor = &(this->_elementFilter);	
}



} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_THRESHOLDING_FILTER_H*/
