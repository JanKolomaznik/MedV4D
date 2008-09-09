/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageConvertor.tcc 
 * @{ 
 **/

#ifndef _IMAGE_CONVERTOR_H
#error File ImageConvertor.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

/*template< typename InputElementType, typename OutputElementType, unsigned OutputDimension, typename Convertor >
void
ConvertImage( const AbstractImage &in, Image< OutputElementType, OutputDimension > );*/

template< typename InputElementType, typename OutputElementType, typename Convertor >
bool
ConvertImage( const AbstractImage &in, Image< OutputElementType, 2 > &out )
{
	
	InputElementType *sPointer1;
	OutputElementType *sPointer2;
	int32 xStride1;
	int32 yStride1;
	int32 xStride2;
	int32 yStride2;
	uint32 height;
	uint32 width;
	sPointer1 = (static_cast< const Image<InputElementType, 2 > & >( in )).GetPointer( width, height, xStride1, yStride1 );
	sPointer2 = out.GetPointer( width, height, xStride2, yStride2 );

	for( uint32 j = 0; j < height; ++j ) {
		InputElementType *pointer1 = sPointer1 + j*yStride1;
		OutputElementType *pointer2 = sPointer2 + j*yStride2;

		for( uint32 i = 0; i < width; ++i ) {
			Convertor::Convert( *pointer1, *pointer2 );
			pointer1 += xStride1;
			pointer2 += xStride2;
		}
	}
	return true;
}

template< typename InputElementType, typename OutputElementType, typename Convertor >
bool
ConvertImage( const AbstractImage &in, Image< OutputElementType, 3 > &out )
{
	
	InputElementType *sPointer1;
	OutputElementType *sPointer2;
	int32 xStride1;
	int32 yStride1;
	int32 zStride1;
	int32 xStride2;
	int32 yStride2;
	int32 zStride2;
	uint32 height;
	uint32 width;
	uint32 depth;

	switch( in.GetDimension() ) {
	case 2: 
		sPointer1 = (static_cast< const Image<InputElementType, 2 > & >( in )).GetPointer( width, height, xStride1, yStride1 );
		depth = 1;
		zStride1 = 0;
		break;
	case 3:	sPointer1 = (static_cast< const Image<InputElementType, 3 > & >( in )).GetPointer( width, height, depth, xStride1, yStride1, zStride1 );
		break;
	default: 
		return false;
	}
	sPointer2 = out.GetPointer( width, height, depth, xStride2, yStride2, zStride2 );

	for( uint32 k = 0; k < depth; ++k ) {
		for( uint32 j = 0; j < height; ++j ) {
			InputElementType *pointer1 = sPointer1 + k*zStride1 + j*yStride1;
			OutputElementType *pointer2 = sPointer2 + k*zStride2 + j*yStride2;

			for( uint32 i = 0; i < width; ++i ) {
				Convertor::Convert( *pointer1, *pointer2 );
				pointer1 += xStride1;
				pointer2 += xStride2;
			}
		}
	}
	return true;
}
//******************************************************************************

template< typename OutputImageType, typename Convertor >
ImageConvertor< OutputImageType, Convertor >
::ImageConvertor( typename ImageConvertor< OutputImageType, Convertor >::Properties  * prop )
	: PredecessorType( prop )
{

}

template< typename OutputImageType, typename Convertor >
ImageConvertor< OutputImageType, Convertor >
::ImageConvertor()
	: PredecessorType( new Properties() )
{
	
}

template< typename OutputImageType, typename Convertor >
bool
ImageConvertor< OutputImageType, Convertor >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	utype = utype;
	D_BLOCK_COMMENT( "++++ Entering ExecutionThreadMethod() - ImageConvertor", "----- Leaving MainExecutionThread() - ImageConvertor" );
	if ( !( _readerBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
		_writerBBox->SetState( MS_CANCELED );
		return false;
	}
	bool result = false;
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
			this->in->GetElementTypeID(), 
			result = ConvertImage< TTYPE, typename ImageTraits< OutputImageType >::ElementType, Convertor >( *(this->in), *(this->out) )
			);
	if( result ) {
		_writerBBox->SetModified();
	} else {
		_writerBBox->SetState( MS_CANCELED );
	}
	return result;
}

template< typename OutputImageType, typename Convertor >
void
ImageConvertor< OutputImageType, Convertor >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

	const unsigned dim = this->in->GetDimension();
	int32 minimums[ ImageTraits<OutputImageType>::Dimension ];
	int32 maximums[ ImageTraits<OutputImageType>::Dimension ];
	float32 voxelExtents[ ImageTraits<OutputImageType>::Dimension ];

	for( unsigned i=0; i < dim; ++i ) {
		const DimensionExtents & dimExt = this->in->GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum;
		voxelExtents[i] = dimExt.elementExtent;
	}
	//fill greater dimensions
	for( unsigned i=dim; i < ImageTraits<OutputImageType>::Dimension; ++i ) {
		minimums[i] = 0;
		maximums[i] = 1;
		voxelExtents[i] = 1.0;
	}

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

template< typename OutputImageType, typename Convertor >
void
ImageConvertor< OutputImageType, Convertor >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	//This kind of filter computes always on whole dataset
	utype = AbstractPipeFilter::RECALCULATION;

	//Image of greater dimension cannot convert to image of lesser dimension
	if( this->in->GetDimension() > ImageTraits< OutputImageType >::Dimension ) {
		throw EDatasetConversionImpossible();
	}
}

template< typename OutputImageType, typename Convertor >
void
ImageConvertor< OutputImageType, Convertor >
::MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype )
{
	utype = utype;

	_readerBBox = this->in->GetWholeDirtyBBox(); 
	_writerBBox = &(this->out->SetWholeDirtyBBox());
}

template< typename OutputImageType, typename Convertor >
void
ImageConvertor< OutputImageType, Convertor >
::AfterComputation( bool successful )
{
	_readerBBox = ReaderBBoxInterface::Ptr();
	_writerBBox = NULL;

	PredecessorType::AfterComputation( successful );
}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_IMAGE_CONVERTOR_H*/

/** @} */

