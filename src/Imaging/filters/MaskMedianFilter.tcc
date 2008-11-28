#ifndef _MASK_MEDIAN_FILTER_H
#error File MaskMedianFilter.tcc cannot be included directly!
#else

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MaskMedianFilter.tcc 
 * @{ 
 **/

namespace Imaging
{


template< unsigned Dim >
MaskMedianFilter2D< Dim >
::MaskMedianFilter2D() : PredecessorType( new Properties() )
{

}

template< unsigned Dim >
MaskMedianFilter2D< Dim >
::MaskMedianFilter2D( typename MaskMedianFilter2D< Dim >::Properties *prop ) 
: PredecessorType( prop ) 
{

}

template< unsigned Dim >
void
MaskMedianFilter2D< Dim >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	if( utype != AbstractPipeFilter::RECALCULATION 
		&& this->_propertiesTimestamp != GetProperties().GetTimestamp() )
	{
		utype = AbstractPipeFilter::RECALCULATION;
	}
}

template< unsigned Dim >
bool
MaskMedianFilter2D< Dim >
::Process2D(
			const ImageRegion< uint8, 2 >	&inRegion,
			ImageRegion< uint8, 2 > 	&outRegion
		 )
{
	if( !this->CanContinue() ) {
		return false;
	}

	ElementType	*inPointer = inRegion.GetPointer();
	int32		i_xStride = inRegion.GetStride( 0 );
	int32		i_yStride = inRegion.GetStride( 1 );
	ElementType	*outPointer = outRegion.GetPointer();
	int32		o_xStride = outRegion.GetStride( 0 );
	int32		o_yStride = outRegion.GetStride( 1 );
	uint32		width = inRegion.GetSize( 0 );
	uint32		height = inRegion.GetSize( 1 );

	int radius = GetProperties().radius;
	int medianOrder = ((2*radius+1) * (2*radius+1)) / 2;

	Histogram histogram;

	ElementType *inRowPointer = inPointer + radius*i_yStride;
	ElementType *outRowPointer = outPointer + radius*o_yStride;
	for( int j =  radius; j < (int)(height - radius); ++j ) {
		ElementType *inElementPointer = inRowPointer + radius*i_xStride;
		ElementType *outElementPointer = outRowPointer + radius*o_xStride;

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

/*
template< unsigned Dim >
bool
MaskMedianFilter2D< Dim >
::Process2D(
			typename MaskMedianFilter2D< Dim >::ElementType	*inPointer,
			int32			i_xStride,
			int32			i_yStride,
			typename MaskMedianFilter2D< Dim >::ElementType	*outPointer,
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

	Histogram histogram;

	ElementType *inRowPointer = inPointer + radius*i_yStride;
	ElementType *outRowPointer = outPointer + radius*o_yStride;
	for( int j =  radius; j < (int)(height - radius); ++j ) {
		ElementType *inElementPointer = inRowPointer + radius*i_xStride;
		ElementType *outElementPointer = outRowPointer + radius*o_xStride;

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
}*/

template< unsigned Dim >
inline typename MaskMedianFilter2D< Dim >::ElementType
MaskMedianFilter2D< Dim >
::GetElementInOrder(
		typename MaskMedianFilter2D< Dim >::Histogram	&histogram,
		uint32						order
	      )
{
	if( histogram.falseCounter < (int32)order ) {
		return TRUE_VALUE;
	}
	return 0;
}

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/


#endif /*_MASK_MEDIAN_FILTER_H*/


