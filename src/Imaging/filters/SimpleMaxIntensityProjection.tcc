#ifndef _SIMPLE_MAX_INTENSITY_PROJECTION_H
#error File SimpleMaxIntensityProjection.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{


template< typename ElementType >
SimpleMaxIntensityProjection< Image< ElementType, 3 > >
::SimpleMaxIntensityProjection() : PredecessorType( new Properties() )
{

}

template< typename ElementType >
SimpleMaxIntensityProjection< Image< ElementType, 3 > >
::SimpleMaxIntensityProjection( typename SimpleMaxIntensityProjection< Image< ElementType, 3 > >::Properties *prop ) 
: PredecessorType( prop ) 
{

}

template< typename ElementType >
bool
SimpleMaxIntensityProjection< Image< ElementType, 3 > >
::ProcessImage(
			const Image< ElementType, 3 > 	&in,
			Image< ElementType, 2 >		&out
		    );
{

	uint32 width1;
	uint32 height1;
	uint32 depth1;
	int32 xStride1;
	int32 yStride1;
	int32 zStride1;
	InputElementType *pointer1 = in.GetPointer( width1, height1, depth1, xStride1, yStride1, zStride1 );

	uint32 width2;
	uint32 height2;
	int32 xStride2;
	int32 yStride2;
	OutputElementType *pointer2 = out.GetPointer( width2, height2, xStride2, yStride2 );

	switch( GetProperties().plane ) {
	case XY_PLANE:
		{
			DoProjection(
				pointer1,
				pointer2,
				xStride1,
				yStride1,
				zStride1,
				xStride2,
				yStride2,
				width1,
				height1,
				depth1,
			    );
		} break;
	case XZ_PLANE:
		{
			DoProjection(
				pointer1,
				pointer2,
				xStride1,
				zStride1,
				yStride1,
				xStride2,
				yStride2,
				width1,
				height1,
				depth1,
			    );
		} break;
	case YZ_PLANE:
		{
			DoProjection(
				pointer1,
				pointer2,
				yStride1,
				zStride1,
				xStride1,
				xStride2,
				yStride2,
				width1,
				height1,
				depth1,
			    );
		} break;
	default:
		ASSERT( false );
	}
}

template< typename ElementType >
void
SimpleMaxIntensityProjection< Image< ElementType, 3 > >
::PrepareOutputDatasets();
{
	PredecessorType::PrepareOutputDatasets();

	int32 minimums[ 2 ];
	int32 maximums[ 2 ];
	float32 voxelExtents[ 2 ];

	switch( GetProperties().plane ) {
	case XY_PLANE:
		{
			minimums[0] = this->in->GetDimensionExtents( 0 ).minimum;
			maximums[0] = this->in->GetDimensionExtents( 0 ).maximum;
			voxelExtents[0] = this->in->GetDimensionExtents( 0 ).elementExtent;

			minimums[1] = this->in->GetDimensionExtents( 1 ).minimum;
			maximums[1] = this->in->GetDimensionExtents( 1 ).maximum;
			voxelExtents[1] = this->in->GetDimensionExtents( 1 ).elementExtent;
		} break;
	case XZ_PLANE:
		{
			minimums[0] = this->in->GetDimensionExtents( 0 ).minimum;
			maximums[0] = this->in->GetDimensionExtents( 0 ).maximum;
			voxelExtents[0] = this->in->GetDimensionExtents( 0 ).elementExtent;

			minimums[1] = this->in->GetDimensionExtents( 2 ).minimum;
			maximums[1] = this->in->GetDimensionExtents( 2 ).maximum;
			voxelExtents[1] = this->in->GetDimensionExtents( 2 ).elementExtent;
		} break;
	case YZ_PLANE:
		{
			minimums[0] = this->in->GetDimensionExtents( 1 ).minimum;
			maximums[0] = this->in->GetDimensionExtents( 1 ).maximum;
			voxelExtents[0] = this->in->GetDimensionExtents( 1 ).elementExtent;

			minimums[1] = this->in->GetDimensionExtents( 2 ).minimum;
			maximums[1] = this->in->GetDimensionExtents( 2 ).maximum;
			voxelExtents[1] = this->in->GetDimensionExtents( 2 ).elementExtent;
		} break;
	default:
		ASSERT( false );
	}

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

template< typename ElementType >
void
SimpleMaxIntensityProjection< Image< ElementType, 3 > >
::DoProjection(
			ElementType	*inPointer,
			ElementType	*outPointer,
			int32		ixStride,
			int32		iyStride,
			int32		izStride,
			int32		oxStride,
			int32		oyStride,
			uint32		width,
			uint32		height,
			uint32		depth
		    )
{

	ElementType *inRowPointer = inPointer;
	ElementType *outRowPointer = outPointer;
	for( uint32 j = 0; j < height; ++j ) {
		ElementType *inColPointer = inRowPointer;
		ElementType *outColPointer = outRowPointer;
		for( uint32 i = 0; i < width; ++i ) {
			ElementType actualPointer = inColPointer;
			ElementType max = *actualPointer;
			
			for( uint32 k = 1; k < depth; ++k ) {
				actualPointer += izStride;
				if( max < *actualPointer ) {
					max = *actualPointer;
				}
			}
			*outColPointer = max;
			inColPointer += ixStride;
			outColPointer += oxStride;
		}
		inRowPointer += iyStride;
		outRowPointer += oyStride;
	}
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_SIMPLE_MAX_INTENSITY_PROJECTION_H*/
