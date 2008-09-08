/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file SimpleProjectionRemote.tcc 
 * @{ 
 **/

#ifndef SIMPLE_PROJ_REMOTE_H
#error File SimpleProjectionRemote.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType>
SimpleProjectionRemote<ImageType>::SimpleProjectionRemote()
: PredecessorType( new Properties() )
{
  AbstractFilterSerializer *ser;  

  // definig vector that will define actual remote pipeline
  FilterSerializerVector m_filterSerializers;

  uint16 filterID = 1;

  // put into the vector serializers instances in order that is in remote pipe
  {
    // insert simpleP serializer
    ser = GeneralFilterSerializer::
      GetFilterSerializer<FID_SimpleProjection>( 
        &m_simpleMIPOptions, filterID++);
    m_filterSerializers.push_back( ser);
  
    // ... for other possible members definig remote pipe filters
  }

  // create job
  m_job = s_cellClient.CreateJob( m_filterSerializers);
}

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType >
void 
SimpleProjectionRemote<ImageType>::PrepareOutputDatasets()
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

/** @} */

