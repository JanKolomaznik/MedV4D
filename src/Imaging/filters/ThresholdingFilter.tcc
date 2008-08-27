/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ThresholdingFilter.tcc 
 * @{ 
 **/

#ifndef _THRESHOLDING_FILTER_H
#error File ThresholdingFilter.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

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

//******************************************************************************
//******************************************************************************

template< typename ImageType >
ThresholdingMaskFilter< ImageType >
::ThresholdingMaskFilter() : PredecessorType( new Properties() )
{
	GetProperties()._functor = &(this->_elementFilter);
}

template< typename ImageType >
ThresholdingMaskFilter< ImageType >
::ThresholdingMaskFilter( typename ThresholdingMaskFilter< ImageType >::Properties *prop ) : PredecessorType( prop )
{
	GetProperties()._functor = &(this->_elementFilter);	
}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_THRESHOLDING_FILTER_H*/

/** @} */

