#ifndef _IMAGE_CONVERTOR_H
#error File ImageConvertor.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

template< typename OutputImageType, typename Convertor = DefaultConvertor >
ImageConvertor< OutputImageType >
::ImageConvertor( ImageConvertor< OutputImageType >::Properties  * prop )
	: PredecessorType( prop )
{

}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
ImageConvertor
::ImageConvertor()
	: PredecessorType( new Properties() )
{
	
}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
bool
ImageConvertor
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{

}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
void
ImageConvertor
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

	const unsigned dim = in->GetDimension();
	size_t minimums[ ImageTraits<OutputImageType>::Dimension ];
	size_t maximums[ ImageTraits<OutputImageType>::Dimension ];
	float32 voxelExtents[ ImageTraits<OutputImageType>::Dimension ];

	for( unsigned i=0; i < dim; ++i ) {
		const DimensionExtents & dimExt = this->in->GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum;
		voxelExtents[i] = dimExt.elementExtent;
	}
	//fill greater dimensions
	for( unsigned i=dim; i < dimImageTraits<OutputImageType>::Dimension; ++i ) {
		minimums[i] = 0;
		maximums[i] = 1;
		voxelExtents[i] = 1.0;
	}

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
void
ImageConvertor
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	//Image of greater dimension cannot convert to image of lesser dimension
	if( this->in->GetDimension() > ImageTraits< OutputImageType >::Dimension ) {
		throw EDatasetConversionImpossible();
	}
}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
void
ImageConvertor
::AfterComputation( bool successful )
{
	PredecessorType::AfterComputation( successful );
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE_CONVERTOR_H*/
