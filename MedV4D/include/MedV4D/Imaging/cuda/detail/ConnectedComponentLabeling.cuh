#ifndef CONNECTED_COMPONENT_LABELING_CUH
#define CONNECTED_COMPONENT_LABELING_CUH

#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"



#define min_valid(a, b) (a < b ? a == 0 ? b : a : b == 0 ? a : b)
__device__ inline uint32
ValidMin( uint32 data[], uint idx, uint syStride, uint szStride )
{
	uint32 value1 = min_valid( data[idx-1], data[idx+1] );
	uint32 value2 = min_valid( data[idx-syStride], data[idx+syStride] );
	uint32 value3 = min_valid( data[idx-szStride], data[idx+szStride] );
	uint32 value = min_valid( value1, value2 );
	return min_valid( value, value3 );
}

__global__ void 
CopyMask( Buffer3D< uint8 > inBuffer, Buffer3D< uint32 > outBuffer );

__global__ void 
InitLut( Buffer3D< uint32 > outBuffer, Buffer1D< uint32 > lut );

//Group equivalence classes
__global__ void 
UpdateLut( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut );

__global__ void 
UpdateLabels( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut );

__global__ void 
ScanImage( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut, int3 blockResolution );



#endif //CONNECTED_COMPONENT_LABELING_CUH
