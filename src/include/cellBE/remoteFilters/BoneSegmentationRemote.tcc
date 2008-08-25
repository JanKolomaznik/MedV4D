#ifndef BONE_SEGMENTATION_FILTER_H
#error File BoneSegmentationRemote.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType>
BoneSegmentationRemote<ImageType>::BoneSegmentationRemote()
: PredecessorType( new Properties() )
{
  AbstractFilterSerializer *ser;  

  // definig vector that will define actual remote pipeline
  FilterSerializerVector m_filterSerializers;

  uint16 filterID = 1;

  // put into the vector serializers instances in order that is in remote pipe
  {
    ser = GeneralFilterSerializer::GetFilterSerializer<Thresholding>( 
      &m_thresholdingOptions, filterID++);
    m_filterSerializers.push_back( ser);
  
    // ... for other possible members definig remote pipe filters
  }

  // create job
  m_job = s_cellClient.CreateJob( m_filterSerializers);
/*
m_job->SendFilterProperties();
  // setting datSets is performed here because actual dataSet may not be created
  // before
  
const ImageType::Ptr inImage = 
		ImageFactory::CreateEmptyImage3DTyped<ElementType>(8, 8, 8);
	for( unsigned i = 0; i < 8; ++i ) {
		for( unsigned j = 0; j < 8; ++j ) {
			for( unsigned k = 0; k < 8; ++k ) {
				inImage->GetElement( i, j, k ) = (i) | (j >> 3) | (k >> 6);
			}
		}
	}

ImageType::Ptr outImage = 
		ImageFactory::CreateEmptyImage3DTyped<ElementType>(8, 8, 8);
	for( unsigned i = 0; i < 8; ++i ) {
		for( unsigned j = 0; j < 8; ++j ) {
			for( unsigned k = 0; k < 8; ++k ) {
				outImage->GetElement( i, j, k ) = (i) | (j >> 3) | (k >> 6);
			}
		}
	}

m_job->SetDataSets( (const AbstractDataSet &) *inImage.get(), (AbstractDataSet &) *outImage.get() );

  m_job->SendDataSetProps();
*/
}

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType >
void 
BoneSegmentationRemote<ImageType>::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

	int32 minimums[M4D::Imaging::ImageTraits<ImageType>::Dimension];
	int32 maximums[M4D::Imaging::ImageTraits<ImageType>::Dimension];
	float32 voxelExtents[M4D::Imaging::ImageTraits<ImageType>::Dimension];

	for( unsigned i=0; i < M4D::Imaging::ImageTraits<ImageType>::Dimension; ++i ) {
		const DimensionExtents & dimExt = this->in->GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum;
		voxelExtents[i] = dimExt.elementExtent;
	}
	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif
