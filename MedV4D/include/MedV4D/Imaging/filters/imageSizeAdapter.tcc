

#ifndef IMAGE_SIZE_ADAPTER_H_
#error File imageSizeAdapter.h cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType>
ImageSizeAdapter<ImageType>
::ImageSizeAdapter( Properties * prop )
	: PredecessorType(prop)
{
	this->_name = "ImageSizeAdapter Filter";
}

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType>
void
ImageSizeAdapter<ImageType>
	::PrepareOutputDatasets(void)
{
	PredecessorType::PrepareOutputDatasets();
	
	Vector<uint32, 3> size(
			this->in->GetDimensionExtents(0).maximum - this->in->GetDimensionExtents(0).minimum,
			this->in->GetDimensionExtents(1).maximum - this->in->GetDimensionExtents(1).minimum,
			this->in->GetDimensionExtents(2).maximum - this->in->GetDimensionExtents(2).minimum
			);
	
	// guarantee that:
	// width and height should be divisible by 32 and at least 32
	if(size[0] < 32) size[0] = 32;
	if(size[1] < 32) size[1] = 32;
	if((size[0] % 32) != 0) size[0] = (1 + (size[0] / 32)) * 32;
	if((size[1] % 32) != 0) size[1] = (1 + (size[1] / 32)) * 32;
	
	_ratio[0] = _ratio[1] = _ratio[2] = 1;
	
	size_t elemsNeeded = size[0] * size[1] * size[2];
	if( (elemsNeeded * sizeof(TOutPixel)) > this->GetProperties().desiredSize)
	{
		// we need to shrink something
		
		// first try to decrease resolution to half
		size[0] = size[1] = (size[0] >> 1);
		elemsNeeded = size[0] * size[1] * size[2];
		_ratio[0] = _ratio[1] = 0.5f;
		
		// if this is not sufficient downsampling even the slice count
		if((elemsNeeded * sizeof(TOutPixel)) > this->GetProperties().desiredSize)
		{
#define SIZEOFSLICE_AFTERDOWNSAMPLIG (sizeof(TOutPixel) * size[0] * size[1])
			// we can afford only desiredSize / SIZEOFSLICE_AFTERDOWNSAMPLIG slices
			size[2] = (this->GetProperties().desiredSize / SIZEOFSLICE_AFTERDOWNSAMPLIG);
			
#define ORIGINAL_DEPTH (this->in->GetDimensionExtents(2).maximum - this->in->GetDimensionExtents(2).minimum)
			_ratio[2] = (float32) size[2] / (float32) ORIGINAL_DEPTH;
		}
	}
	
	const unsigned dim = this->in->GetDimension();
	int32 minimums[dim];
	int32 maximums[dim];
	float32 voxelExtents[dim];

	for( unsigned i=0; i < dim; i++ ) {
		minimums[i] = 0;
		maximums[i] = size[i];
		voxelExtents[i] = this->in->GetDimensionExtents( i ).elementExtent;
	}

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType>
bool
ImageSizeAdapter<ImageType>
	::ProcessImage(const ImageType &in, OutImageType &out)
{		
	switch(in.Dimension)
	{
		case 2:
			break;
			
		case 3:
			Process3DImage(this);
			break;
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType>
void
ImageSizeAdapter<ImageType>
	::Process3DImage(Self *self)
{
	//typedef NearestNeighborInterpolator<ImageType> InterpolatorType;
	typedef LinearInterpolator<ImageType> InterpolatorType;
	InterpolatorType interpolator(self->in);
		
	typename OutImageType::PointType outStrides;
	typename OutImageType::SizeType outSize;
	TOutPixel *outDataPointer = 
		self->out->GetPointer(outSize, outStrides);
	
	typename InterpolatorType::CoordType coord;
	
	Vector<float32, ImageType::Dimension> inverseRatio(
			1 / _ratio[0],
			1 / _ratio[1],
			1 / _ratio[2]
			           );
	
	for(uint32 z = 0; z < outSize[2]; z++)
	{
		for(uint32 y=0; y < outSize[1]; y++)
		{
			for(uint32 x=0; x < outSize[0]; x++)
			{
				coord[0] = x * inverseRatio[0];
				coord[1] = y * inverseRatio[1];
				coord[2] = z * inverseRatio[2];
				*outDataPointer = interpolator.Get(coord);
				outDataPointer++;
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

#endif