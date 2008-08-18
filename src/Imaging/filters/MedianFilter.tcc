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
::MedianFilter2D( typename MedianFilter2D< Image< InputElementType, 3 > >::Properties *prop ) 
: PredecessorType( prop ) 
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
	if( !this->CanContinue() ) {
		return false;
	}

	int radius = GetProperties().radius;
	int medianOrder = ((2*radius+1) * (2*radius+1)) / 2;

	std::map< InputElementType, int > histogram;
	for( int j = y1 + radius; j < (y2 - radius); ++j ) {
		//initialize histogram
		histogram.clear();
		for( int l = j-radius; l <= j+radius; ++l ){
			for( int k = x1; k <= x1+(2*radius)+1; ++k ){
				++(histogram[ in.GetElement( k, l, slice ) ]);
			}
		}
		out.GetElement( x1 + radius, j, slice ) = GetElementInOrder( histogram, medianOrder );


		for( int i = x1 + radius + 1; i < (x2 - radius); ++i ) {
			for( int k = -radius; k <= radius; ++k ){
				--(histogram[ in.GetElement( i-(radius+1), j + k, slice ) ]);
				++(histogram[ in.GetElement( i+radius, j + k, slice ) ]);
				out.GetElement( i, j, slice ) = GetElementInOrder( histogram, medianOrder );
			}
		}
	}
	return true;
}

template< typename InputElementType >
inline InputElementType
MedianFilter2D< Image< InputElementType, 3 > >
::GetElementInOrder(
		MedianFilter2D< Image< InputElementType, 3 > >::Histogram	&histogram,
		uint32								order
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
	throw 10;
	return (InputElementType)0;
}

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_MEDIAN_FILTER_H*/
