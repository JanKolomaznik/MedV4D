#ifndef CONNECTED_COMPONENT_LABELING_CUH
#define CONNECTED_COMPONENT_LABELING_CUH

#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"

__device__ int lutUpdated;

#define min_valid(a, b) (a < b ? a == 0 ? b : a : b == 0 ? a : b)
__device__ uint32
ValidMin( uint32 data[], uint idx, uint syStride, uint szStride )
{
	uint32 value1 = min_valid( data[idx-1], data[idx+1] );
	uint32 value2 = min_valid( data[idx-syStride], data[idx+syStride] );
	uint32 value3 = min_valid( data[idx-szStride], data[idx+szStride] );
	uint32 value = min_valid( value1, value2 );
	return min_valid( value, value3 );
}

__global__ void 
CopyMask( Buffer3D< uint8 > inBuffer, Buffer3D< uint32 > outBuffer )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < inBuffer.mLength ) {
		outBuffer.mData[idx] = inBuffer.mData[idx]!=0 ? idx+1 : 0;
	}
}

__global__ void 
InitLut( Buffer3D< uint32 > outBuffer, Buffer1D< uint32 > lut )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < outBuffer.mLength ) {
		lut.mData[idx] = outBuffer.mData[idx];// = outBuffer.mData[idx] != 0 ? idx+1 : 0;
	}
}

//Group equivalence classes
__global__ void 
UpdateLut( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;
	uint32 label, ref;
	if ( idx < buffer.mLength ) {
		label = buffer.mData[idx];

		if (label == idx+1) {		
			ref = label-1;
			label = lut.mData[idx];
			while (ref != label-1) {
				ref = label-1;
				label = lut.mData[ref];
			}
			lut.mData[idx] = label;
		}
	}
}

__global__ void 
UpdateLabels( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < buffer.mLength ) {
		uint label = buffer.mData[idx];
		if ( label > 0 ) {
			buffer.mData[idx] = lut.mData[label-1];
		}
	}
}

__global__ void 
ScanImage( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut, int3 blockResolution )
{
	__shared__ uint32 data[MAX_SHARED_MEMORY];
	
	const int cBlockDim = 8;
	const int cRadius = 1;
	const uint syStride = cBlockDim+2*cRadius;
	const uint szStride = (cBlockDim+2*cRadius) * (cBlockDim+2*cRadius);

	uint sidx = (threadIdx.y+cRadius) * syStride + (threadIdx.z+cRadius) * szStride + threadIdx.x + cRadius;
	int3 size = toInt3( buffer.mSize );
	int3 blockCoordinates = GetBlockCoordinates( blockResolution, __mul24(blockIdx.y, gridDim.x) + blockIdx.x );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	
	
	int3 coordinates = blockOrigin + threadIdx;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), size );
	int idx = IdxFromCoordStrides( coordinates, buffer.mStrides );

	FillSharedMemory3D_8x8x8< uint32, cRadius, syStride, szStride >( data, sidx, buffer.mData, buffer.mStrides, size, blockOrigin, coordinates, idx );

	__syncthreads();

	if( !projected ) {
		uint32 current = data[sidx];
		if ( current != 0 ) {
			uint32 minLabel = ValidMin( data, sidx, syStride, szStride );
			if ( minLabel < current && minLabel != 0) {
				lut.mData[current-1] = minLabel < lut.mData[current-1] ? minLabel : lut.mData[current-1];
				lutUpdated = 1;
			}
		}
	}
}



#endif //CONNECTED_COMPONENT_LABELING_CUH
