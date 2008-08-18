#ifndef _MEDIAN_FILTER_H
#error File MedianFilter.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{


template< typename InputElementType >
MedianFilter2D< Image< InputElementType, 3 > >
::MedianFilter2D() : PredecessorType( new Properties() )
{

}

template< typename InputElementType >
MedianFilter2D< Image< InputElementType, 3 > >
::MedianFilter2D( ConvolutionFilter2D< Image< InputElementType, 3 >, MatrixElement >::Properties *prop ) 
: PredecessorType( prop ), 
{

}

template< typename InputElementType >
bool
MedianFilter2D< Image< InputElementType, 3 > >
::ProcessSlice(	
			const Image< InputElementType, 3 > 	&in,
			Image< InputElementType, 3 >		&out,
			int32			x1,	
			int32			y1,	
			int32			x2,	
			int32			y2,	
			int32			slice
		    )
{
	uint32 radius = GetProperties().radius;
	for( int32 j = y1 + radius; j < y2 - radius; ++j ) {
		for( int32 i = x1 + radius; i < x2 - radius; ++i ) {
			out.GetElement( i, j, slice ) = GetMedian( radius, in, i, j, slice );
		}
	}
}

template< typename InputElementType >
InputElementType
MedianFilter2D< Image< InputElementType, 3 > >
::ProcessSlice(
		uint32					radius,
		const Image< InputElementType, 3 > 	&in,
		int32					x,
		int32					y,
		int32					slice
	      )
{
	return in.GetElement( x, y, slice );
}

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_MEDIAN_FILTER_H*/
