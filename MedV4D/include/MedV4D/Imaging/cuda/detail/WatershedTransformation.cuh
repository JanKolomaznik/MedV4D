#ifndef WATERSHED_TRANSFORMATION_CUH
#define WATERSHED_TRANSFORMATION_CUH





template< typename TElement >
struct RegionBorderDetection3DFtor
{
	typedef typename TypeTraits< TElement >::SignedClosestType SignedElement;

	RegionBorderDetection3DFtor(): radius( make_int3( 1, 1, 1 ) )
	{}

	__device__ uint8
	operator()( TElement data[], uint idx, uint syStride, uint szStride, uint gIdx )
	{
		TElement val = data[idx];
		if ( val != data[idx-1] || val != data[idx-syStride] || val != data[idx-szStride] ) {
			return 255;
		} else {
			return 0;
		}
	}

	int3 radius;
};


template< typename TEType >
__global__ void 
InitWatershedBuffers( Buffer3D< uint32 > labeledRegionsBuffer, Buffer3D< TEType > tmpBuffer, TEType infinity )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < tmpBuffer.mLength ) {
		tmpBuffer.mData[idx] = labeledRegionsBuffer.mData[idx] == 0 ? infinity : 0;
	}
}


template< typename TInEType, typename TTmpEType >
__global__ void 
WShedEvolution( Buffer3D< uint32 > labeledRegionsBuffer, Buffer3D< TInEType > inputBuffer, Buffer3D< TTmpEType > tmpBuffer, int3 blockResolution, TTmpEType infinity )
{
	__shared__ uint32 labels[MAX_SHARED_MEMORY];
	__shared__ TTmpEType tmpValues[MAX_SHARED_MEMORY];
	
	int3 radius = make_int3(1,1,1);
	uint syStride = blockDim.x+2*radius.x;
	uint szStride = (blockDim.x+2*radius.x) * (blockDim.y+2*radius.y);

	uint3 size = inputBuffer.mSize;
	int3 strides = inputBuffer.mStrides;
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int3 blockCoordinates = GetBlockCoordinates ( blockResolution, blockId );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	int3 coordinates = blockOrigin;
	//uint tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
	coordinates.x += threadIdx.x;
	coordinates.y += threadIdx.y;
	coordinates.z += threadIdx.z;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), make_int3( size.x, size.y, size.z ) );
	
	int idx = coordinates.x * strides.x + coordinates.y * strides.y + coordinates.z * strides.z;
	uint sidx = (threadIdx.y+radius.y) * syStride + (threadIdx.z+radius.z) * szStride + threadIdx.x + radius.x;
	labels[sidx] = labeledRegionsBuffer.mData[ idx ];
	tmpValues[sidx] = tmpBuffer.mData[ idx ];
	
	uint3 sIdx;
	int3 mCoordinates = blockOrigin;
	switch( threadIdx.z ) {
	case 0:
		sIdx.x = threadIdx.x + radius.x;
		sIdx.y = threadIdx.y + radius.y;
		sIdx.z = 0;
		break;
	case 1:
		sIdx.x = threadIdx.x + radius.x;
		sIdx.y = threadIdx.y + radius.y;
		sIdx.z = blockDim.z + radius.z;
		break;
	case 2:
		sIdx.x = threadIdx.x + radius.x;
		sIdx.y = 0;
		sIdx.z = threadIdx.y + radius.z;
		break;
	case 3:
		sIdx.x = threadIdx.x + radius.x;
		sIdx.y = blockDim.y + radius.y;
		sIdx.z = threadIdx.y + radius.z;
		break;
	case 4:
		sIdx.x = 0;
		sIdx.y = threadIdx.y + radius.y;
		sIdx.z = threadIdx.x + radius.z;
		break;
	case 5:
		sIdx.x = blockDim.x + radius.x;
		sIdx.y = threadIdx.y + radius.y;
		sIdx.z = threadIdx.x + radius.z;
		break;
	case 6:
		if ( threadIdx.y < 4 ) {
			sIdx.x = threadIdx.x + radius.x;
			sIdx.y = (threadIdx.y & 1)*(blockDim.y + radius.y);
			sIdx.z = (threadIdx.y >> 1)*(blockDim.z + radius.z);
		} else {
			sIdx.x = ((threadIdx.y-4) >> 1)*(blockDim.x + radius.x);
			sIdx.y = threadIdx.x + radius.x;
			sIdx.z = (threadIdx.y & 1)*(blockDim.z + radius.z);
		}
		break;
	case 7:
		if ( threadIdx.y < 4 ) {
			sIdx.x = (threadIdx.y & 1)*(blockDim.x + radius.x);
			sIdx.y = ((threadIdx.y) >> 1)*(blockDim.y + radius.y);
			sIdx.z = threadIdx.x + radius.z;
		} else {	
			sIdx.x = threadIdx.x < 4 ? 0 : (blockDim.x + radius.x);
			sIdx.y = (threadIdx.x >> 1) & 1 ? 0 : (blockDim.y + radius.y);
			sIdx.z = threadIdx.x & 1 ? 0 : (blockDim.z + radius.z);
		}
		break;
	default:
		break;
	}
	mCoordinates.x += sIdx.x - radius.x;
	mCoordinates.y += sIdx.y - radius.y;
	mCoordinates.z += sIdx.z - radius.z;
	ProjectionToInterval( mCoordinates, make_int3(0,0,0), make_int3( size.x, size.y, size.z ) );

	labels[sIdx.y*syStride + sIdx.z*szStride + sIdx.x] = labeledRegionsBuffer.mData[ mCoordinates.x * strides.x + mCoordinates.y * strides.y + mCoordinates.z * strides.z ];
	tmpValues[sIdx.y*syStride + sIdx.z*szStride + sIdx.x] = tmpBuffer.mData[ mCoordinates.x * strides.x + mCoordinates.y * strides.y + mCoordinates.z * strides.z ];

	__syncthreads();

	if( !projected ) {
		int minIdx = -1;
		TInEType value = inputBuffer.mData[ idx ];
		TTmpEType minVal = max( tmpValues[ sidx ] - value,TTmpEType(0) );
		for ( int i = sidx-1; i <= sidx+1; ++i ) {
			for ( int j = i-syStride; j <= i+syStride; j+=syStride ) {
				for ( int k = j-szStride; k <= j+szStride; k+=szStride ) {
					if( tmpValues[ k ] < minVal ) {
						minVal = tmpValues[ k ];
						minIdx = k;
					}
				}
			}
		}
		if( minIdx != -1 ) {
			labeledRegionsBuffer.mData[ idx ] = labels[ minIdx ];
			tmpBuffer.mData[ idx ] = tmpValues[minIdx] + value;
			wshedUpdated = 1;
		}
		/*
		for( unsigned it = 0; it < 2; ++it ) {
			TTmpEType minVal = max( tmpValues[ sidx ] - value,TTmpEType(0) );
			for ( int i = sidx-1; i <= sidx+1; ++i ) {
				for ( int j = i-syStride; j <= i+syStride; j+=syStride ) {
					for ( int k = j-szStride; k <= j+szStride; k+=szStride ) {
						if( tmpValues[ k ] < minVal ) {
							minVal = tmpValues[ k ];
							minIdx = k;
						}
					}
				}
			}
			if( minIdx != -1 ) {
				labels[ sidx ] = labels[ minIdx ];
				tmpValues[ sidx ] = tmpValues[minIdx] + value;
				wshedUpdated = 1;
			}
		}
		labeledRegionsBuffer.mData[ idx ] = labels[ sidx ];
		tmpBuffer.mData[ idx ] = tmpValues[sidx];
		*/		
	}
}

#endif //WATERSHED_TRANSFORMATION_CUH