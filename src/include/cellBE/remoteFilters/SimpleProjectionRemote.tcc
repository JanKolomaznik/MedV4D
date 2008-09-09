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
	CellBE::AbstractFilterSerializer *ser;  

	// definig vector that will define actual remote pipeline
	CellBE::FilterSerializerVector m_filterSerializers;

	uint16 filterID = 1;

	// put into the vector serializers instances in order that is in remote pipe
	{
		// insert simpleP serializer
		ser = CellBE::GeneralFilterSerializer::
		GetFilterSerializer<SimpleProjection<ImageType> >( 
		&m_simpleProjOptions, filterID++);
		m_filterSerializers.push_back( ser);

		// ... for other possible members definig remote pipe filters
	}

	// create job
	this->m_job = this->s_cellClient.CreateJob( m_filterSerializers);
}

///////////////////////////////////////////////////////////////////////////////

template< typename ImageType >
void 
SimpleProjectionRemote<ImageType>::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

	int32 minimums[ 2 ];
	int32 maximums[ 2 ];
	float32 voxelExtents[ 2 ];

	switch( GetPlane() ) {
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

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif

/** @} */

