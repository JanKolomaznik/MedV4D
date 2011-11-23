

__kernel void PseudoRes(const __global float *srcData, __global float *dstData, int4 size) {
	int gid = get_global_id(0);

	int x, y;
	x = (get_global_id(0) % (size.x-2)) + 1;
	y = (get_global_id(0) / (size.x-2)) + 1;
	
	int zstride = size.x * size.y;
	int ystride = size.x;
	
	int4 sizeDest;
	sizeDest.x = size.x-2;
	sizeDest.y = size.y-2;
	sizeDest.z = size.z-2;

	int zstrideDest = sizeDest.x * sizeDest.y;
	int ystrideDest = sizeDest.x;

	float sqrt6div7 = sqrt(6.0f/7.0f);
	float onediv6 = 1.0f/6.0f;
	
	int planeOffset = y * size.x + x;

	int z;
	for(z = 1; z < size.z - 1; z++) {
		float accum = 0;
		int workOffset = zstride * z + planeOffset;
		
		accum += srcData[workOffset - 1];
		accum += srcData[workOffset + 1];
		accum += srcData[workOffset - ystride];
		accum += srcData[workOffset + ystride];
		accum += srcData[workOffset - zstride];
		accum += srcData[workOffset + zstride];
		
		dstData[zstrideDest * (z-1) + ystrideDest * (y-1) + x - 1] = sqrt6div7 * (srcData[workOffset] - onediv6 * accum);
	}
}

__kernel void LocMeanAndVar(const __global float *srcData, __global float *dstMean, __global float *dstVar, 
							int radius, int4 size) {
	//int gid = get_global_id(0);

/*	int startOffset = gid * size.x;
	int y, z;
	z = gid / size.y;
	y = gid % size.y;*/
	int z = get_group_id(0);
	int y = get_local_id(0);
	
	int zstride = size.x * size.y;
	int ystride = size.x;

	int startOffset = z * zstride + y * ystride;

	int ymin = max(0, y-radius) - y;
	int ymax = min(size.y-1, y+radius) - y;
	int zmin = max(0, z-radius) - z;
	int zmax = min(size.z-1, z+radius) - z;
	
	int x;
	for(x = 0; x < size.x; x++) {
		int dstOffset = startOffset + x;
		int xmin = max(0, x-radius) - x;
		int xmax = min(size.x-1, x+radius) - x;
		
		int count = 0;
		float accum = 0;
		for(int k = zmin; k <= zmax; k++) { 
			for(int j = ymin; j <= ymax; j++) {
				int walkOffset = dstOffset + (k) * zstride + (j) * ystride; 
				for(int i = xmin; i <= xmax; i++) {
					accum += srcData[walkOffset+i];
					count++;
				}
			}
		}
		float mean = accum / (float)count;

		accum = 0;
		for(int k = zmin; k <= zmax; k++) { 
			for(int j = ymin; j <= ymax; j++) {
				int walkOffset = dstOffset + (k) * zstride + (j) * ystride; 
				for(int i = xmin; i <= xmax; i++) {
					float diff = mean - srcData[walkOffset+i];
					accum += diff * diff;
				}
			}
		}
		
		dstMean[dstOffset] = mean;
		dstVar[dstOffset] = (accum / (float)count);
	}
}

void getNeighbourhood(const __global float *srcData, float *dstNbh, int nbh, int dstOffset, int ystride, int zstride, int4 size) {
	int idxNbh = 0;
	int i,j,k;
	for(k = -nbh; k <= nbh; k++)
	for(j = -nbh; j <= nbh; j++) {
		int tmpOffset = dstOffset + k * zstride + j * ystride;
		for(i = -nbh; i <= nbh; i++) {
			dstNbh[idxNbh++] = srcData[tmpOffset + i];
		}
	}
}

float L2NormSq(float *array1, float *array2, int size) {
	float accum = 0;

	for(int i = 0; i < size; i++) {
		float val = array1[i];// - array2[i];
		accum += val * val;
	}
	return accum;
}

float getNeighbourhoodL2NormSq(const __global float *srcData, int nbh, int offset1, int offset2, int ystride, int zstride) {
	int i,j,k;
	float accum = 0;
	for(k = -nbh; k <= nbh; k++)
	for(j = -nbh; j <= nbh; j++) {
		int tmpOffset = k * zstride + j * ystride;
		int loadOffset1 = tmpOffset + offset1;
		int loadOffset2 = tmpOffset + offset2;
		for(i = -nbh; i <= nbh; i++) {
			float val = srcData[loadOffset1 + i] - srcData[loadOffset2 + i];
			accum += val * val;
		}
	}
	return accum;
}

// HAS LIMIT:
//  neighbourhood <= 2
//  radius <= 5
__kernel void NLMeansOpt(const __global float *srcData, 
					const __global float *srcMean, const __global float *srcVar, 
					__global float *dstData, 
					float variance, float weightConst,
					float meanAccept, float sigma, int radius, int neighbourhood,
					int4 size) {
	int gid = get_global_id(0);
	
	int startOffset = gid * size.x;
	int y, z;
	z = gid / size.y;
	y = gid % size.y;
	
	int zstride = size.x * size.y;
	int ystride = size.x;
	
	int nbhsize = neighbourhood * 2 + 1;
	nbhsize = nbhsize * nbhsize	* nbhsize;

	int border = 1 + radius + neighbourhood;
	if (y < border || y >= size.y - border || z < border || z >= size.z - border) {
		return;
	}

	float weights[1331]; // max (5*2+1)ˆ3
	float values[1331]; // max (5*2+1)ˆ3

	for(int x = border; x < size.x - border; x++) {
		int dstOffset = startOffset + x;
//		float targetNbh[125]; // max (2*2+1)ˆ3
//		float workNbh[125]; // max (2*2+1)ˆ3
		int i,j,k;
		// get neighbourhood of target voxel
		//getNeighbourhood(srcData, targetNbh, neighbourhood, dstOffset, x, y, z, ystride, zstride, size);
		
		// get parameters of target voxel
		float mean1, var1;
		mean1 = srcMean[dstOffset];
		var1 = srcVar[dstOffset];
		
		//compute weights
		int computedValues = 0;
		int kmin = max(0,z - radius), kmax = min(size.z-1,z+radius);
		int jmin = max(0,y - radius), jmax = min(size.y-1,y+radius);
		int imin = max(0,x - radius), imax = min(size.x-1,x+radius);
		
		float sigmaRcp = 1./sigma;
		for(k = kmin; k <= kmax; k++)
		for(j = jmin; j <= jmax; j++) {
			int tmpOffset = k * zstride + j * ystride;
			for(i = imin; i <= imax; i++) {
				int workOffset = tmpOffset + i;
				float mean2 = srcMean[workOffset];
				float var2 = srcVar[workOffset];
				float var1divvar2 = var1 / var2;

				// check if voxel has similar params
				if((fabs(mean1 - mean2) <= variance * meanAccept) &&
					(sigma < var1divvar2 && var1divvar2 < sigmaRcp)) {
					 //get neighborhood
					//getNeighbourhood(srcData, workNbh, neighbourhood, workOffset, i, j, k, ystride, zstride, size);
					 // compute weight and store value
					float L2coeffSq = getNeighbourhoodL2NormSq(srcData, neighbourhood, dstOffset, workOffset, ystride, zstride);
					weights[computedValues] = exp(- L2coeffSq / weightConst);
					values[computedValues] = srcData[workOffset];
					computedValues++;
				}
			}
		}
		
		// normalize weights
		float sumWeights = 0;
		for(i = 0; i < computedValues; i++) {
			sumWeights += weights[i];
		}

		float result = 0;
		for(i = 0; i < computedValues; i++) {
			result += values[i] * weights[i];
		}
		
		dstData[dstOffset] = result / sumWeights;
	}
} 

#define BLOCK_NLOPT		13

#define BLOCK_WEIGHTS	9

// radius <= 4
// neighbourhood <= 2
__kernel void NLMeansV2(const __global float *srcData, 					 
					__global float *dstData, 
					float weightConst,
					int radius, int neighbourhood,
					int4 size,
					__local float locData[BLOCK_NLOPT][BLOCK_NLOPT][BLOCK_NLOPT],
					__local float locWeights[BLOCK_WEIGHTS][BLOCK_WEIGHTS][BLOCK_WEIGHTS],
					__local float locAccWeights[BLOCK_WEIGHTS][BLOCK_WEIGHTS],
					__local float locOneDimWeights[BLOCK_WEIGHTS]) 
{
	int gidX = get_group_id(0);
	int gidY = get_group_id(1);

	int lidX = get_local_id(0);
	int lidY = get_local_id(1);

	int zstride = size.x * size.y;
	int ystride = size.x;

	int nbhsize = neighbourhood * 2 + 1;
	nbhsize = nbhsize * nbhsize	* nbhsize;
	int border = radius + neighbourhood;
	int influenceSize = border * 2 + 1; // influence of each voxel area

	int x, y, z; // top left corner of the view window in source/destination data
	x = gidX + lidX;
	y = gidY + lidY;

	int dstX, dstY;	// coords of target processed voxel in source/destination data
	dstX = gidX + border;
	dstY = gidY + border;

	// initial data load
	for(z = 0; z < influenceSize; z++) {
		locData[z][lidY][lidX] = srcData[zstride * z + ystride * y + x];
	}

	int center = border;

	int locWeightX = lidX - neighbourhood;
	int locWeightY = lidY - neighbourhood;
	
	barrier(CLK_LOCAL_MEM_FENCE);

	for(z = 0; z < size.z - 2*border; z++) {
		int dstZ = z + border;

		//compute weights
		////////////////////////////////////////////
		float accum = 0;
		if(lidX >= neighbourhood && lidX < BLOCK_NLOPT - neighbourhood &&
			lidY >= neighbourhood && lidY < BLOCK_NLOPT - neighbourhood)
		{
			for(int lidZ = neighbourhood; lidZ < BLOCK_NLOPT - neighbourhood; lidZ++) 
			{
				//compute NL weights for each voxel that is being considered
				float L2coeffSq = 0;
				for(int k = -neighbourhood + 1; k <= neighbourhood - 1; k++) {
					for(int j = -neighbourhood + 1; j <= neighbourhood - 1; j++) {
						for(int i = -neighbourhood + 1; i <= neighbourhood - 1; i++) {
							float diff = locData[center+k][center+j][center+i] - locData[lidZ+k][lidY+j][lidX+i];
							L2coeffSq += diff*diff;
						}
					}
				}
				float weight = exp(- L2coeffSq / weightConst);
				//float weight = ( L2coeffSq / weightConst);
				locWeights[lidZ - neighbourhood][locWeightY][locWeightX] = weight;
				accum += weight;
			}

			// store accum weights
			locAccWeights[locWeightY][locWeightX] = accum;
		}

		// sum weights
		//////////////////////////////////////////////
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// sum 2D into 1D
		int maxSumIdx = 2*radius + 1;
		accum = 0;
		if(lidY == 0) {
			for(int j = 0; j< maxSumIdx; j++) {
				accum += locAccWeights[j][locWeightX];
			}
			
			locOneDimWeights[locWeightX] = accum;			
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// sum 1D into total weight
		accum = 0;
		if(lidX == 0 && lidY == 0) {
			for(int i= 0; i< maxSumIdx; i++) {
				accum += locOneDimWeights[i];
			}
			// distribute accumulated weight into other threads
			locOneDimWeights[0] = accum;
		}

		// get accumulated weight
		barrier(CLK_LOCAL_MEM_FENCE);
		accum = locOneDimWeights[0];
		float recpWeightSum = 1.0f/accum;

		// compute final color
		///////////////////////////////////////////////
		if(lidX >= neighbourhood && lidX < BLOCK_NLOPT - neighbourhood &&
			lidY >= neighbourhood && lidY < BLOCK_NLOPT - neighbourhood)
		{
			float accum = 0;
			for(int locWeightZ = 0; locWeightZ <= 2*radius; locWeightZ++) 
			{
				//compute NL weights for each voxel that is being considered
				float weight = locWeights[locWeightZ][locWeightY][locWeightX];
				float val = locData[locWeightZ+neighbourhood][locWeightY+neighbourhood][locWeightX+neighbourhood];
				accum += weight * val;
			}
			locAccWeights[locWeightY][locWeightX] = accum;
		}

		// sum 2D array of values into 1D
		barrier(CLK_LOCAL_MEM_FENCE);

		accum = 0;
		if(lidY == 0) {
			for(int j = 0; j< maxSumIdx; j++) {
				accum += locAccWeights[j][locWeightX];
			}
			
			locOneDimWeights[locWeightX] = accum;			
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// sum 1D into total value
		accum = 0;
		if(lidX == 0 && lidY == 0) {
			for(int i= 0; i< maxSumIdx; i++) {
				accum += locOneDimWeights[i];
			}
			locOneDimWeights[0] = accum;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(lidX == 0 && lidY == 0)
			dstData[zstride * dstZ + ystride * dstY + dstX] = locOneDimWeights[0] * recpWeightSum;

		// shift local data one voxel higher if performing next plane
		barrier(CLK_LOCAL_MEM_FENCE); 
		if(z < size.z - 2*border - 1) { 
			int tmpZ;
			for(tmpZ = 0; tmpZ < influenceSize-1; tmpZ++) {
				locData[tmpZ][lidY][lidX] = locData[tmpZ + 1][lidY][lidX];
			}
			// read new slice
			locData[influenceSize-1][lidY][lidX] = srcData[zstride * (z + influenceSize) + ystride * y + x];
		}
	}
}

// radius <= 4
// neighbourhood <= 2
__kernel void NLMeansOptV2(const __global float *srcData, 					 
					__global float *dstData, 
					__global float *srcMean, 
					__global float *srcVar, 
					int radius, int neighbourhood,
					float weightConst,
					float meanAccept,
					float variance,
					float sigma,
					int4 size,
					__local float locData[BLOCK_NLOPT][BLOCK_NLOPT][BLOCK_NLOPT],
					__local float locWeights[BLOCK_WEIGHTS][BLOCK_WEIGHTS][BLOCK_WEIGHTS],
					__local float locAccWeights[BLOCK_WEIGHTS][BLOCK_WEIGHTS],
					__local float locOneDimWeights[BLOCK_WEIGHTS]) 
{
	int gidX = get_group_id(0);
	int gidY = get_group_id(1);

	int lidX = get_local_id(0);
	int lidY = get_local_id(1);

	int zstride = size.x * size.y;
	int ystride = size.x;

	int nbhsize = neighbourhood * 2 + 1;
	nbhsize = nbhsize * nbhsize	* nbhsize;
	int border = radius + neighbourhood;
	int influenceSize = border * 2 + 1; // influence of each voxel area

	int x, y, z; // top left corner of the view window in source/destination data
	x = gidX + lidX;
	y = gidY + lidY;

	int dstX, dstY;	// coords of target processed voxel in source/destination data
	dstX = gidX + border;
	dstY = gidY + border;

	// initial data load
	for(z = 0; z < influenceSize; z++) {
		locData[z][lidY][lidX] = srcData[zstride * z + ystride * y + x];
	}

	int center = border;

	int locWeightX = lidX - neighbourhood;
	int locWeightY = lidY - neighbourhood;
	
	barrier(CLK_LOCAL_MEM_FENCE);

	for(z = 0; z < size.z - 2*border; z++) {
		int dstZ = z + border;

		// get parameters of target voxel
		float mean1, var1;
		mean1 = srcMean[zstride * (z+center) + ystride * (y+center) + (x+center)];
		var1 = srcVar[zstride * (z+center) + ystride * (y+center) + (x+center)];
		float sigmaRcp = 1./sigma;

		//compute weights
		////////////////////////////////////////////
		float accum = 0;
		if(lidX >= neighbourhood && lidX < BLOCK_NLOPT - neighbourhood &&
			lidY >= neighbourhood && lidY < BLOCK_NLOPT - neighbourhood)
		{
			for(int lidZ = neighbourhood; lidZ < BLOCK_NLOPT - neighbourhood; lidZ++) 
			{
				//compute NL weights for each voxel that is being considered
				 
				float mean2 = srcMean[zstride * (z+lidZ) + ystride * (y+lidY) + (x+lidX)];
				float var2 = srcVar[zstride * (z+lidZ) + ystride * (y+lidY) + (x+lidX)];
				float var1divvar2 = var1 / var2;
				
				float weight = 0;

				if(/*(fabs(mean1 - mean2) <= variance * meanAccept) &&*/
					(sigma < var1divvar2 && var1divvar2 < sigmaRcp)) {
					float L2coeffSq = 0;
					for(int k = -neighbourhood + 1; k <= neighbourhood - 1; k++) {
						for(int j = -neighbourhood + 1; j <= neighbourhood - 1; j++) {
							for(int i = -neighbourhood + 1; i <= neighbourhood - 1; i++) {
								float diff = locData[center+k][center+j][center+i] - locData[lidZ+k][lidY+j][lidX+i];
								L2coeffSq += diff*diff;
							}
						}
					}
					weight = exp(- L2coeffSq / weightConst);
				}
				//weight = (lidX == center && lidY == center && lidZ == center)?1:0;

				locWeights[lidZ - neighbourhood][locWeightY][locWeightX] = weight;
				accum += weight;
			}

			// store accum weights
			locAccWeights[locWeightY][locWeightX] = accum;
		}

		// sum weights
		//////////////////////////////////////////////
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// sum 2D into 1D
		int maxSumIdx = 2*radius + 1;
		accum = 0;
		if(lidY == 0) {
			for(int j = 0; j< maxSumIdx; j++) {
				accum += locAccWeights[j][locWeightX];
			}
			
			locOneDimWeights[locWeightX] = accum;			
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// sum 1D into total weight
		accum = 0;
		if(lidX == 0 && lidY == 0) {
			for(int i= 0; i< maxSumIdx; i++) {
				accum += locOneDimWeights[i];
			}
			// distribute accumulated weight into other threads
			locOneDimWeights[0] = accum;
		}

		// get accumulated weight
		barrier(CLK_LOCAL_MEM_FENCE);
		accum = locOneDimWeights[0];
		float recpWeightSum = 1.0f/accum;

		// compute final color
		///////////////////////////////////////////////
		if(lidX >= neighbourhood && lidX < BLOCK_NLOPT - neighbourhood &&
			lidY >= neighbourhood && lidY < BLOCK_NLOPT - neighbourhood)
		{
			float accum = 0;
			for(int locWeightZ = 0; locWeightZ <= 2*radius; locWeightZ++) 
			{
				//compute NL weights for each voxel that is being considered
				float weight = locWeights[locWeightZ][locWeightY][locWeightX];
				float val = locData[locWeightZ+neighbourhood][locWeightY+neighbourhood][locWeightX+neighbourhood];
				accum += weight * val;
			}
			locAccWeights[locWeightY][locWeightX] = accum;
		}

		// sum 2D array of values into 1D
		barrier(CLK_LOCAL_MEM_FENCE);

		accum = 0;
		if(lidY == 0) {
			for(int j = 0; j< maxSumIdx; j++) {
				accum += locAccWeights[j][locWeightX];
			}
			
			locOneDimWeights[locWeightX] = accum;			
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// sum 1D into total value
		accum = 0;
		if(lidX == 0 && lidY == 0) {
			for(int i= 0; i< maxSumIdx; i++) {
				accum += locOneDimWeights[i];
			}
			locOneDimWeights[0] = accum;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(lidX == 0 && lidY == 0)
			dstData[zstride * dstZ + ystride * dstY + dstX] = locOneDimWeights[0] * recpWeightSum;

		// shift local data one voxel higher if performing next plane
		barrier(CLK_LOCAL_MEM_FENCE); 
		if(z < size.z - 2*border - 1) { 
			int tmpZ;
			for(tmpZ = 0; tmpZ < influenceSize-1; tmpZ++) {
				locData[tmpZ][lidY][lidX] = locData[tmpZ + 1][lidY][lidX];
			}
			// read new slice
			locData[influenceSize-1][lidY][lidX] = srcData[zstride * (z + influenceSize) + ystride * y + x];
		}
	}
}