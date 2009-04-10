

#ifndef DECIMATION_H_
#error File decimation.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType, typename InterpolatorType>
DecimationFilter<ImageType, InterpolatorType>
::DecimationFilter( Properties * prop )
	: PredecessorType(prop)
{
	this->_name = "Decimator Filter";
}

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType, typename InterpolatorType>
void
DecimationFilter<ImageType, InterpolatorType>
	::PrepareOutputDatasets(void)
{
	PredecessorType::PrepareOutputDatasets();
	
	float32 ratio = this->GetProperties().ratio;
		
	const unsigned dim = this->in->GetDimension();
	int32 minimums[ ImageTraits<ImageType>::Dimension ];
	int32 maximums[ ImageTraits<ImageType>::Dimension ];
	float32 voxelExtents[ ImageTraits<ImageType>::Dimension ];

	for( unsigned i=0; i < dim-1; i++ ) {
		const DimensionExtents & dimExt = this->in->GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum * ratio;
		voxelExtents[i] = dimExt.elementExtent * ratio;
	}
	// 3-rd dim maximus is the same since we don't change slice count
	const DimensionExtents & dimExt = this->in->GetDimensionExtents( 2 );
	minimums[2] = dimExt.minimum;
	maximums[2] = dimExt.maximum;
	voxelExtents[2] = dimExt.elementExtent * ratio;	// but voxel extends we change

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType, typename InterpolatorType>
bool
DecimationFilter<ImageType, InterpolatorType>
	::ProcessImage(const ImageType &in, ImageType &out)
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

template< typename ImageType, typename InterpolatorType>
void
DecimationFilter<ImageType, InterpolatorType>
	::Process3DImage(Self *self)
{
	InterpolatorType interpolator(self->in);
		
	typename ImageType::PointType outStrides;
	typename ImageType::SizeType outSize;
	typename ImageType::Element *outDataPointer = 
		self->out->GetPointer(outSize, outStrides);
	
	typename InterpolatorType::CoordType coord;
	
	uint32 backCoef = 1 / self->GetProperties().ratio;
	
	for(uint32 z = 0; z < outSize[2]; z++)
	{
		for(uint32 y=0; y < outSize[1]; y++)
		{
			for(uint32 x=0; x < outSize[0]; x++)
			{
				coord[0] = x * backCoef;
				coord[1] = y * backCoef;
				coord[2] = z;
				*outDataPointer = interpolator.Get(coord);
				outDataPointer++;
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/


#endif