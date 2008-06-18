#ifndef _EXAMPLE_IMAGE_FILTERS_H
#error File ExampleImageFilters.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename InputElementType, typename OutputElementType >
CopyImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::CopyImageFilter()
{

}

template< typename InputElementType, typename OutputElementType >
void
CopyImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< OutputElementType, 3 >		&out,
			size_t			x1,	
			size_t			y1,	
			size_t			x2,	
			size_t			y2,	
			size_t			slice
		    )
{
	for( size_t i = x1; i < x2; ++i ) {
		for( size_t j = y1; j < y2; ++j ) {
			out.GetElement( i, j, slice ) = in.GetElement( i, j, slice );
		}
	}
}

template< typename InputElementType, typename OutputElementType >
void
CopyImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::PrepareOutputDatasets()
{
	//TODO - improve
	const Image< InputElementType, 3 > &in = this->GetInputImage();
	size_t minimums[3];
	size_t maximums[3];
	float32 voxelExtents[3];

	for( unsigned i=0; i < 3; ++i ) {
		const DimensionExtents & dimExt = in.GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum;
		voxelExtents[i] = dimExt.elementExtent;
	}
	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

//*****************************************************************************
//*****************************************************************************

template< typename ElementType >
ColumnMaxImageFilter< ElementType >
::ColumnMaxImageFilter()
{

}

template< typename ElementType >
void
ColumnMaxImageFilter< ElementType >
::ProcessVolume(
			const Image< ElementType, 3 > 		&in,
			Image< ElementType, 2 >			&out,
			size_t					x1,
			size_t					y1,
			size_t					z1,
			size_t					x2,
			size_t					y2,
			size_t					z2
		    )
{
	for( size_t i = x1; i < x2; ++i ) {
		for( size_t j = y1; j < y2; ++j ) {
			ElementType max = in.GetElement( i, j, z1 );
			for( size_t k = z1+1; k < z2; ++k ) {
				if( in.GetElement( i, j, k ) > max ) {
					max = in.GetElement( i, j, k );
				}
			}
			out.GetElement( i, j ) = max;
		}
	}
}

template< typename ElementType >
void
ColumnMaxImageFilter< ElementType >
::PrepareOutputDatasets()
{
	//TODO - improve
	const Image< ElementType, 3 > &in = this->GetInputImage();
	size_t minimums[2];
	size_t maximums[2];
	float32 pixelExtents[2];

	for( unsigned i=0; i < 2; ++i ) {
		const DimensionExtents & dimExt = in.GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum;
		pixelExtents[i] = dimExt.elementExtent;
	}
	this->SetOutputImageSize( minimums, maximums, pixelExtents );
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_BASIC_IMAGE_FILTERS_H*/
