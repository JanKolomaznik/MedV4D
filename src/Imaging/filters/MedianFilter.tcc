/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MedianFilter.tcc 
 * @{ 
 **/

#ifndef _MEDIAN_FILTER_H
#error File MedianFilter.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{


template< typename InputImageType >
MedianFilter2D< InputImageType >
::MedianFilter2D() : PredecessorType( new Properties() )
{

}

template< typename InputImageType >
MedianFilter2D< InputImageType >
::MedianFilter2D( typename MedianFilter2D< InputImageType >::Properties *prop ) 
: PredecessorType( prop ) 
{

}

template< typename InputImageType >
void
MedianFilter2D< InputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	if( utype != AbstractPipeFilter::RECALCULATION 
		&& this->_propertiesTimestamp != GetProperties().GetTimestamp() )
	{
		utype = AbstractPipeFilter::RECALCULATION;
	}
}

template< typename InputImageType >
bool
MedianFilter2D< InputImageType >
::Process2D(
			typename ImageTraits< InputImageType >::ElementType	*inPointer,
			int32			i_xStride,
			int32			i_yStride,
			typename ImageTraits< InputImageType >::ElementType	*outPointer,
			int32			o_xStride,
			int32			o_yStride,
			uint32			width,
			uint32			height
		 )
{
	if( !this->CanContinue() ) {
		return false;
	}

	int radius = GetProperties().radius;
	int medianOrder = ((2*radius+1) * (2*radius+1)) / 2;

	std::map< InputElementType, int > histogram;

	InputElementType *inRowPointer = inPointer + radius*i_yStride;
	InputElementType *outRowPointer = outPointer + radius*o_yStride;
	for( int j =  radius; j < (int)(height - radius); ++j ) {
		InputElementType *inElementPointer = inRowPointer + radius*i_xStride;
		InputElementType *outElementPointer = outRowPointer + radius*o_xStride;

		//initialize histogram
		histogram.clear();
		for( int l = -radius; l <= radius; ++l ){
			for( int k = -radius; k <= radius; ++k ){
				++(histogram[  *(inElementPointer + k*i_xStride + l*i_yStride) ]);
			}
		}
		*outElementPointer = GetElementInOrder( histogram, medianOrder );


		for( int i = radius + 1; i < (int)(width - radius); ++i ) {
			inElementPointer += i_xStride;
			outElementPointer += o_xStride;

			for( int k = -radius; k <= radius; ++k ){
				--(histogram[ *(inElementPointer - (radius+1)*i_xStride + k*i_yStride) ]);
				++(histogram[ *(inElementPointer + radius*i_xStride + k*i_yStride) ]);
				*outElementPointer = GetElementInOrder( histogram, medianOrder );
			}
		}
		inRowPointer += i_yStride;
		outRowPointer += o_yStride;
	}
	return true;
}

template< typename InputImageType >
inline typename ImageTraits< InputImageType >::ElementType
MedianFilter2D< InputImageType >
::GetElementInOrder(
		typename MedianFilter2D< InputImageType >::Histogram	&histogram,
		uint32						order
	      )
{
	uint32 count = 0;
	typename Histogram::iterator it = histogram.begin();

	while( it != histogram.end() && (count += it->second) < order ) {
		++it;
	}
	if( it !=histogram.end() ) {
		return it->first;
	}
	return (typename ImageTraits< InputImageType >::ElementType)0;
}

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_MEDIAN_FILTER_H*/

/** @} */

