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
{
  AbstractFilterSerializer *ser;  

  // definig vector that will define actual remote pipeline
  FilterSerializerVector m_filterSerializers;

  // put into the vector serializers instances in order that is in remote pipe
  {
    ser = GeneralFilterSerializer::GetFilterSerializer( 
      &m_thresholdingOptions );
    m_filterSerializers.push_back( ser);
  
    // ... for other possible members definig remote pipe filters
  }

  // create dataSetSerializers for input & output dataSets
  {
  m_inSerializer = GeneralDataSetSerializer::GetDataSetSerializer<ImageType>(
    getInput().GetDataSet());
  m_outSerializer = GeneralDataSetSerializer::GetDataSetSerializer<ImageType>(
    getOutPut().GetDataSet());  
  }

  // create job
  m_job = m_cellClient.CreateJob( 
    m_filterSerializers, inSerializer, outSerializer);
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
