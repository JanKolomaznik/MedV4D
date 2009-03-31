/**
 * @ingroup imaging 
 * @author Szabolcs Grof
 * @file ImageTransform.tcc 
 * @{ 
 **/

#ifndef IMAGE_TRANSFORM_H_
#error File ImageTransform.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< typename ElementType >
bool
TransformImage( const Image< ElementType, 2 > &in, Image< ElementType, 2 > &out, AbstractFilter::Properties* &properties, InterpolatorBase< Image< ElementType, 2 > >* &interpolator )
{
	
	ElementType *sPointer;
	int32 xStride;
	int32 yStride;
	int32 height;
	int32 width;
	typename ImageTransform< ElementType, 2 >::Properties* prop = dynamic_cast< typename ImageTransform< ElementType, 2 >::Properties* >( properties );
	typedef typename InterpolatorBase< Image< ElementType, 2 > >::CoordType CoordType;
	sPointer = out.GetPointer( (uint32&)width, (uint32&)height, xStride, yStride );
	Vector< CoordType, 2 > RotationMatrix(
					CoordType( std::cos( -prop->_rotation[0] ), -std::sin( -prop->_rotation[0] ) ),
					CoordType( std::sin( -prop->_rotation[0] ),  std::cos( -prop->_rotation[0] ) )
					);

	for( int32 j = 0; j < (int32)height; ++j ) {
		ElementType *pointer = sPointer + j*yStride;

		for( int32 i = 0; i < (int32)width; ++i ) {
			CoordType point( ( i - width/2 ) * RotationMatrix[0][0] + ( j - height/2 ) * RotationMatrix[0][1] + width/2 - prop->_translation[0],
					 ( i - width/2 ) * RotationMatrix[1][0] + ( j - height/2 ) * RotationMatrix[1][1] + height/2 - prop->_translation[1] );

			if ( point[0] < 0 ) point[0] = 0;
			if ( point[0] >= width ) point[0] = width - 1;
			if ( point[1] < 0 ) point[1] = 0;
			if ( point[1] >= height ) point[1] = height - 1;

			*pointer = interpolator->Get( point ); 
			pointer += xStride;
		}
	}
	return true;
}

template< typename ElementType >
bool
TransformImage( const Image< ElementType, 3 > &in, Image< ElementType, 3 > &out, AbstractFilter::Properties* &properties, InterpolatorBase< Image< ElementType, 3 > >* &interpolator )
{
	
	ElementType *sPointer;
	int32 xStride;
	int32 yStride;
	int32 zStride;
	int32 height;
	int32 width;
	int32 depth;
	typename ImageTransform< ElementType, 3 >::Properties* prop = dynamic_cast< typename ImageTransform< ElementType, 3 >::Properties* >( properties );

	Vector< uint32, 3 > size;
	Vector< int32, 3 > strides;
	typedef typename InterpolatorBase< Image< ElementType, 3 > >::CoordType CoordType;
	sPointer = out.GetPointer( size, strides );
	width = (int32)size[0];
	height = (int32)size[1];
	depth = (int32)size[2];
	xStride = strides[0];
	yStride = strides[1];
	zStride = strides[2];

	Vector< CoordType, 3 > RotationMatrixX(
					CoordType( 1.0, 0.0, 0.0 ),
					CoordType( 0.0, std::cos( -prop->_rotation[0] ), -std::sin( -prop->_rotation[0] ) ),
					CoordType( 0.0, std::sin( -prop->_rotation[0] ),  std::cos( -prop->_rotation[0] ) )
					);

	Vector< CoordType, 3 > RotationMatrixY(
					CoordType( std::cos( -prop->_rotation[1] ), 0.0, -std::sin( -prop->_rotation[1] ) ),
					CoordType( 0.0, 1.0, 0.0 ),
					CoordType( std::sin( -prop->_rotation[1] ), 0.0,  std::cos( -prop->_rotation[1] ) )
					);

	Vector< CoordType, 3 > RotationMatrixZ(
					CoordType( std::cos( -prop->_rotation[2] ), -std::sin( -prop->_rotation[2] ), 0.0 ),
					CoordType( std::sin( -prop->_rotation[2] ),  std::cos( -prop->_rotation[2] ), 0.0 ),
					CoordType( 0.0, 0.0, 1.0 )
					);

	for( int32 k = 0; k < depth; ++k ) {
		for( int32 j = 0; j < height; ++j ) {
			ElementType *pointer = sPointer + k*zStride + j*yStride;

			for( int32 i = 0; i < (int32)width; ++i ) {
				CoordType point( ( i - width/2 ) * RotationMatrixX[0][0] + ( j - height/2 ) * RotationMatrixX[0][1] + ( k - depth/2 ) * RotationMatrixX[0][2] + width/2, 
						 ( i - width/2 ) * RotationMatrixX[1][0] + ( j - height/2 ) * RotationMatrixX[1][1] + ( k - depth/2 ) * RotationMatrixX[1][2] + height/2,
						 ( i - width/2 ) * RotationMatrixX[2][0] + ( j - height/2 ) * RotationMatrixX[2][1] + ( k - depth/2 ) * RotationMatrixX[2][2] + depth/2 );
				//std::cout<<( i - (int32)width/2 ) * RotationMatrixX[0][0] + ( j - (int32)height/2 ) * RotationMatrixX[0][1] + ( k - (int32)depth/2 ) * RotationMatrixX[0][2] + (int32)width/2<<" "<<i<<" "<<j<<" "<<k<<" "<<point[0]<<" "<<point[1]<<" "<<point[2]<<std::endl;

				CoordType  tmp( ( point[0] - width/2 ) * RotationMatrixY[0][0] + ( point[1] - height/2 ) * RotationMatrixY[0][1] + ( point[2] - depth/2 ) * RotationMatrixY[0][2] + width/2,
						( point[0] - width/2 ) * RotationMatrixY[1][0] + ( point[1] - height/2 ) * RotationMatrixY[1][1] + ( point[2] - depth/2 ) * RotationMatrixY[1][2] + height/2,
						( point[0] - width/2 ) * RotationMatrixY[2][0] + ( point[1] - height/2 ) * RotationMatrixY[2][1] + ( point[2] - depth/2 ) * RotationMatrixY[2][2] + depth/2 );

				point = tmp;
				//std::cout<<i<<" "<<j<<" "<<k<<" "<<point[0]<<" "<<point[1]<<" "<<point[2]<<std::endl;
				tmp = CoordType( ( point[0] - width/2 ) * RotationMatrixZ[0][0] + ( point[1] - height/2 ) * RotationMatrixZ[0][1] + ( point[2] - depth/2 ) * RotationMatrixZ[0][2] + width/2,
						 ( point[0] - width/2 ) * RotationMatrixZ[1][0] + ( point[1] - height/2 ) * RotationMatrixZ[1][1] + ( point[2] - depth/2 ) * RotationMatrixZ[1][2] + height/2,
						 ( point[0] - width/2 ) * RotationMatrixZ[2][0] + ( point[1] - height/2 ) * RotationMatrixZ[2][1] + ( point[2] - depth/2 ) * RotationMatrixZ[2][2] + depth/2 );

				point = tmp;
				//std::cout<<i<<" "<<j<<" "<<k<<" "<<point[0]<<" "<<point[1]<<" "<<point[2]<<std::endl;
				//std::cout<<"here"<<std::endl;
				point[0] -= prop->_translation[0];
				point[1] -= prop->_translation[1];
				point[2] -= prop->_translation[2];

				if ( point[0] < 0 ) point[0] = 0;
				if ( point[0] >= width ) point[0] = width - 1;
				if ( point[1] < 0 ) point[1] = 0;
				if ( point[1] >= height ) point[1] = height - 1;
				if ( point[2] < 0 ) point[2] = 0;
				if ( point[2] >= depth ) point[2] = depth - 1;

				*pointer = interpolator->Get( point );
				
				pointer += xStride;
			}
		}
	}
	return true;
}

//***************************************************************************************************************

template< typename ElementType, uint32 dim >
ImageTransform< ElementType, dim >
::ImageTransform( typename ImageTransform< ElementType, dim >::Properties  * prop )
	: PredecessorType( prop )
{
	this->_name = "ImageTransform";
}

template< typename ElementType, uint32 dim >
ImageTransform< ElementType, dim >
::ImageTransform()
	: PredecessorType( new Properties() )
	
{
	this->_name = "ImageTransform";
}

template< typename ElementType, uint32 dim >
bool
ImageTransform< ElementType, dim >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	utype = utype;
	if ( !( _readerBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
		_writerBBox->SetState( MS_CANCELED );
		return false;
	}
	bool result = false;
	InterpolatorBase< ImageType > *interpolator = new LinearInterpolator< ImageType >( this->in );
	result = TransformImage< ElementType >( *(this->in), *(this->out), this->_properties, interpolator );
	delete interpolator;
	if( result ) {
		_writerBBox->SetModified();
	} else {
		_writerBBox->SetState( MS_CANCELED );
	}
	return result;
}

template< typename ElementType, uint32 dim >
void
ImageTransform< ElementType, dim >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

	const unsigned dimension = this->in->GetDimension();
	int32 minimums[ ImageTraits<ImageType>::Dimension ];
	int32 maximums[ ImageTraits<ImageType>::Dimension ];
	float32 voxelExtents[ ImageTraits<ImageType>::Dimension ];

	for( unsigned i=0; i < dimension; ++i ) {
		const DimensionExtents & dimExt = this->in->GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum;
		voxelExtents[i] = dimExt.elementExtent * dynamic_cast< Properties* >( this->_properties )->_scale[i];
	}
	//fill greater dimensions
	for( unsigned i=dimension; i < ImageTraits<ImageType>::Dimension; ++i ) {
		minimums[i] = 0;
		maximums[i] = 1;
		voxelExtents[i] = 1.0;
	}

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

template< typename ElementType, uint32 dim >
void
ImageTransform< ElementType, dim >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	//This kind of filter computes always on whole dataset
	utype = AbstractPipeFilter::RECALCULATION;

	//Image of greater dimension cannot convert to image of lesser dimension
	if( this->in->GetDimension() > ImageTraits< ImageType >::Dimension ) {
		throw EDatasetTransformImpossible();
	}
}

template< typename ElementType, uint32 dim >
void
ImageTransform< ElementType, dim >
::MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype )
{
	utype = utype;

	_readerBBox = this->in->GetWholeDirtyBBox(); 
	_writerBBox = &(this->out->SetWholeDirtyBBox());
}

template< typename ElementType, uint32 dim >
void
ImageTransform< ElementType, dim >
::AfterComputation( bool successful )
{
	_readerBBox = ReaderBBoxInterface::Ptr();
	_writerBBox = NULL;

	PredecessorType::AfterComputation( successful );
}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*IMAGE_TRANSFORM_H_*/

/** @} */

