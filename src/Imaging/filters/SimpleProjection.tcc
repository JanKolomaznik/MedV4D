/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file SimpleProjection.tcc 
 * @{ 
 **/

#ifndef _SIMPLE_PROJECTION_H
#error File SimpleProjection.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{


template< typename ElementType >
SimpleProjection< Image< ElementType, 3 > >
::SimpleProjection() : PredecessorType( new Properties() )
{

}

template< typename ElementType >
SimpleProjection< Image< ElementType, 3 > >
::SimpleProjection( typename SimpleProjection< Image< ElementType, 3 > >::Properties *prop ) 
: PredecessorType( prop ) 
{

}

template< typename ElementType >
void
SimpleProjection< Image< ElementType, 3 > >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	if( this->_propertiesTimestamp != GetProperties().GetTimestamp() )
	{
		utype = AbstractPipeFilter::RECALCULATION;
		this->_callPrepareOutputDatasets = true;
	}
}

template< typename ElementType >
bool
SimpleProjection< Image< ElementType, 3 > >
::ProcessImage(
			const Image< ElementType, 3 > 	&in,
			Image< ElementType, 2 >		&out
		    )
{
	switch( GetProjectionType() ) {
	case PT_MAX:
		return ProcessImageHelper< MaxOperator< ElementType > >( in, out );
	case PT_SUM:
		return ProcessImageHelper< SumOperator< ElementType > >( in, out );
	case PT_AVERAGE:
		return ProcessImageHelper< AverageOperator< ElementType > >( in, out );
	default:
		ASSERT( false );
	}
}

template< typename ElementType >
template< typename OperatorType >
bool
SimpleProjection< Image< ElementType, 3 > >
::ProcessImageHelper(
			const Image< ElementType, 3 > 	&in,
			Image< ElementType, 2 >		&out
		    )
{

	Vector< uint32, 3 > size1;
	Vector< int32, 3 > strides1;
/*	uint32 width1;
	uint32 height1;
	uint32 depth1;
	int32 xStride1;
	int32 yStride1;
	int32 zStride1;*/
	ElementType *pointer1 = in.GetPointer( size1, strides1 );

	/*uint32 width2;
	uint32 height2;
	int32 xStride2;
	int32 yStride2;*/
	Vector< uint32, 2 > size2;
	Vector< int32, 2 > strides2;
	ElementType *pointer2 = out.GetPointer( size2, strides2 );

	switch( GetProperties().plane ) {
	case XY_PLANE:
		{
			return DoProjection< OperatorType >(
				pointer1,
				pointer2,
				strides1[0],
				strides1[1],
				strides1[2],
				strides2[0],
				strides2[1],
				size1[0],
				size1[1],
				size1[2]
			    );
		} break;
	case XZ_PLANE:
		{
			return DoProjection< OperatorType >(
				pointer1,
				pointer2,
				strides1[0],
				strides1[2],
				strides1[1],
				strides2[0],
				strides2[1],
				size1[0],
				size1[2],
				size1[1]
			    );
		} break;
	case YZ_PLANE:
		{
			return DoProjection< OperatorType >(
				pointer1,
				pointer2,
				strides1[1],
				strides1[2],
				strides1[0],
				strides2[0],
				strides2[1],
				size1[1],
				size1[2],
				size1[0]
			    );
		} break;
	default:
		ASSERT( false );
	}
	return false;
}

template< typename ElementType >
void
SimpleProjection< Image< ElementType, 3 > >
::PrepareOutputDatasets()
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
template< typename OperatorType >
bool
SimpleProjection< Image< ElementType, 3 > >
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
	if( !this->CanContinue() ) {
		return false;
	}
	
	OperatorType funcOperator;

	ElementType *inRowPointer = inPointer;
	ElementType *outRowPointer = outPointer;
	for( uint32 j = 0; j < height; ++j ) {
		ElementType *inColPointer = inRowPointer;
		ElementType *outColPointer = outRowPointer;
		for( uint32 i = 0; i < width; ++i ) {
			ElementType *actualPointer = inColPointer;
			funcOperator.Init( *actualPointer );
			
			for( uint32 k = 1; k < depth; ++k ) {
				actualPointer += izStride;
				funcOperator.AddElement( *actualPointer );
			}
			*outColPointer = funcOperator.Result();
			inColPointer += ixStride;
			outColPointer += oxStride;
		}
		inRowPointer += iyStride;
		outRowPointer += oyStride;
	}
	
	return true;
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_SIMPLE_PROJECTION_H*/

/** @} */

