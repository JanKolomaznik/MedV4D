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
ConvertImage( const AImage &in, Image< OutputElementType, OutputDimension > );*/

template< typename InputElementType, typename OutputElementType, typename Convertor >
bool
ConvertImage( const AImage &in, Image< OutputElementType, 2 > &out )
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
ConvertImage( const AImage &in, Image< OutputElementType, 3 > &out )
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

		Vector< uint32, 2 > size2d;
		Vector< int32, 2 > strides2d;
		Vector< uint32, 3 > size3d;
		Vector< int32, 3 > strides3d;

	switch( in.GetDimension() ) {
	case 2: 
		sPointer1 = (static_cast< const Image<InputElementType, 2 > & >( in )).GetPointer( size2d, strides2d );
		width = size2d[0];
		height = size2d[1];
		depth = 1;

		xStride1 = strides2d[0];
		yStride1 = strides2d[1];
		zStride1 = 0;
		break;
	case 3:	
		sPointer1 = (static_cast< const Image<InputElementType, 3 > & >( in )).GetPointer( size3d, strides3d );
		width = size3d[0];
		height = size3d[1];
		depth = size3d[2];

		xStride1 = strides3d[0];
		yStride1 = strides3d[1];
		zStride1 = strides3d[2];
		break;
	default: 
		return false;
	}
	Vector< uint32, 3 > size;
	Vector< int32, 3 > strides;
	sPointer2 = out.GetPointer( size, strides );
	width = size[0];
	height = size[1];
	depth = size[2];
	xStride2 = strides[0];
	yStride2 = strides[1];
	zStride2 = strides[2];



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
	this->_name = "ImageConvertor";
}

template< typename OutputImageType, typename Convertor >
ImageConvertor< OutputImageType, Convertor >
::ImageConvertor()
	: PredecessorType( new Properties() )
{
	this->_name = "ImageConvertor";
}

template< typename OutputImageType, typename Convertor >
bool
ImageConvertor< OutputImageType, Convertor >
::ExecutionThreadMethod( APipeFilter::UPDATE_TYPE utype )
{
	utype = utype;
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
::BeforeComputation( APipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	//This kind of filter computes always on whole dataset
	utype = APipeFilter::RECALCULATION;

	//Image of greater dimension cannot convert to image of lesser dimension
	if( this->in->GetDimension() > ImageTraits< OutputImageType >::Dimension ) {
		throw EDatasetConversionImpossible();
	}
}

template< typename OutputImageType, typename Convertor >
void
ImageConvertor< OutputImageType, Convertor >
::MarkChanges( APipeFilter::UPDATE_TYPE utype )
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

