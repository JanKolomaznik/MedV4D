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
TransformImage( const Image< ElementType, 2 > &in, Image< ElementType, 2 > &out, AbstractFilter::Properties* properties, InterpolatorBase< Image< ElementType, 2 > >* interpolator, int32 sliceNum )
{
	
	ElementType *sPointer;
	int32 xStride;
	int32 yStride;
	int32 height, oldheight;
	int32 width, oldwidth;
	float32 xExtent, yExtent;

	typename ImageTransform< ElementType, 2 >::Properties* prop = dynamic_cast< typename ImageTransform< ElementType, 2 >::Properties* >( properties );
	typedef typename InterpolatorBase< Image< ElementType, 2 > >::CoordType CoordType;
	sPointer = out.GetPointer( (uint32&)width, (uint32&)height, xStride, yStride );
	oldwidth = (int32)(width / prop->_sampling[0]);
	oldheight = (int32)(height / prop->_sampling[1]);
	xExtent = out.GetDimensionExtents( 0 ).elementExtent;
	yExtent = out.GetDimensionExtents( 1 ).elementExtent;

	Vector< CoordType, 2 > RotationMatrix(
					CoordType( std::cos( -prop->_rotation[0] ), -std::sin( -prop->_rotation[0] ) ),
					CoordType( std::sin( -prop->_rotation[0] ),  std::cos( -prop->_rotation[0] ) )
					);

	for( int32 j = 0; j < (int32)height; ++j ) {
		ElementType *pointer = sPointer + j*yStride;

		for( int32 i = 0; i < (int32)width; ++i ) {
			CoordType point( ( ( i - width/2 ) * RotationMatrix[0][0] * xExtent + ( j - height/2 ) * RotationMatrix[0][1] * yExtent ) / xExtent + width/2,
					 ( ( i - width/2 ) * RotationMatrix[1][0] * xExtent + ( j - height/2 ) * RotationMatrix[1][1] * yExtent ) / yExtent + height/2 );

			point[0] -= prop->_translation[0];
			point[1] -= prop->_translation[1];

			point[0] = ( point[0] / prop->_scale[0] + ( width / 2 - width / ( 2 * prop->_scale[0] ) ) );
			point[1] = ( point[1] / prop->_scale[1] + ( height / 2 - height / ( 2 * prop->_scale[1] ) ) );

			point[0] /= prop->_sampling[0];
			point[1] /= prop->_sampling[1];

			if ( ( point[0] < 0 ) |
			     ( point[0] > ( oldwidth - 1 ) ) |
			     ( point[1] < 0 ) |
			     ( point[1] > ( oldheight - 1 ) ) ) *pointer = 0;

			else *pointer = interpolator->Get( point );

			pointer += xStride;
		}
	}
	return true;
}

template< typename ElementType >
class TransformSlice
{
public:
	TransformSlice( const Image< ElementType, 3 > &input, Image< ElementType, 3 > &output, typename ImageTransform< ElementType, 3 >::Properties* properties, InterpolatorBase< Image< ElementType, 3 > >* interp, int32 sNum )
		: in( input ), out( output ), prop( properties ), interpolator( interp ), sliceNum( sNum )
	{}
	void operator()()
	{
	
		ElementType *sPointer, *pointer;
		int32 xStride;
		int32 yStride;
		int32 zStride;
		int32 height, oldheight;
		int32 width, oldwidth;
		int32 depth, olddepth;
		float32 xExtent, yExtent, zExtent;
		float32 newwidth, newheight, newdepth;

		Vector< uint32, 3 > size;
		Vector< int32, 3 > strides;
		sPointer = out.GetPointer( size, strides );
		typedef typename InterpolatorBase< Image< ElementType, 3 > >::CoordType CoordType;
		width = (int32)size[0];
		height = (int32)size[1];
		depth = (int32)size[2];
		oldwidth = (int32)(width / prop->_sampling[0]);
		oldheight = (int32)(height / prop->_sampling[1]);
		olddepth = (int32)(depth / prop->_sampling[2]);
		xStride = strides[0];
		yStride = strides[1];
		zStride = strides[2];
		xExtent = out.GetDimensionExtents( 0 ).elementExtent;
		yExtent = out.GetDimensionExtents( 1 ).elementExtent;
		zExtent = out.GetDimensionExtents( 2 ).elementExtent;
		newwidth = width * xExtent;
		newheight = height * yExtent;
		newdepth = depth * zExtent;
		

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

		int32 k = sliceNum;
		if ( k >= depth ) return;
		for( int32 j = 0; j < height; ++j ) {

			pointer = sPointer + k*zStride + j*yStride;

			for( int32 i = 0; i < (int32)width; ++i ) {

				CoordType point( ( xExtent * i - newwidth/2 ) * RotationMatrixX[0][0] + ( yExtent * j - newheight/2 ) * RotationMatrixX[0][1] + ( zExtent * k - newdepth/2 ) * RotationMatrixX[0][2], 
						 ( xExtent * i - newwidth/2 ) * RotationMatrixX[1][0] + ( yExtent * j - newheight/2 ) * RotationMatrixX[1][1] + ( zExtent * k - newdepth/2 ) * RotationMatrixX[1][2],
						 ( xExtent * i - newwidth/2 ) * RotationMatrixX[2][0] + ( yExtent * j - newheight/2 ) * RotationMatrixX[2][1] + ( zExtent * k - newdepth/2 ) * RotationMatrixX[2][2] );

				CoordType  tmp( point[0] * RotationMatrixY[0][0] + point[1] * RotationMatrixY[0][1] + point[2] * RotationMatrixY[0][2],
						point[0] * RotationMatrixY[1][0] + point[1] * RotationMatrixY[1][1] + point[2] * RotationMatrixY[1][2],
						point[0] * RotationMatrixY[2][0] + point[1] * RotationMatrixY[2][1] + point[2] * RotationMatrixY[2][2] );

				point = tmp;

				tmp = CoordType( point[0] * RotationMatrixZ[0][0] + point[1] * RotationMatrixZ[0][1] + point[2] * RotationMatrixZ[0][2] + newwidth/2,
						 point[0] * RotationMatrixZ[1][0] + point[1] * RotationMatrixZ[1][1] + point[2] * RotationMatrixZ[1][2] + newheight/2,
						 point[0] * RotationMatrixZ[2][0] + point[1] * RotationMatrixZ[2][1] + point[2] * RotationMatrixZ[2][2] + newdepth/2 );

				point = CoordType ( tmp[0] / xExtent, tmp[1] / yExtent, tmp[2] / zExtent );

				point[0] -= prop->_translation[0];
				point[1] -= prop->_translation[1];
				point[2] -= prop->_translation[2];

				point[0] = ( point[0] / prop->_scale[0] + ( width / 2 - width / ( 2 * prop->_scale[0] ) ) );
				point[1] = ( point[1] / prop->_scale[1] + ( height / 2 - height / ( 2 * prop->_scale[1] ) ) );
				point[2] = ( point[2] / prop->_scale[2] + ( depth / 2 - depth / ( 2 * prop->_scale[2] ) ) );

				point[0] /= prop->_sampling[0];
				point[1] /= prop->_sampling[1];
				point[2] /= prop->_sampling[2];

				if ( ( point[0] < 0 ) |
				     ( point[0] > ( oldwidth - 1 ) ) |
				     ( point[1] < 0 ) |
			   	     ( point[1] > ( oldheight - 1 ) ) |
			   	     ( point[2] < 0 ) |
			   	     ( point[2] > ( olddepth - 1 ) ) ) *pointer = 0;

				else *pointer = interpolator->Get( point );
			
				pointer += xStride;
			}
		}
	}
private:
	const Image< ElementType, 3 > &in;
	Image< ElementType, 3 > &out;
	typename ImageTransform< ElementType, 3 >::Properties* prop;
	InterpolatorBase< Image< ElementType, 3 > >* interpolator;
	int32 sliceNum;
	
};

template< typename ElementType >
bool
TransformImage( const Image< ElementType, 3 > &in, Image< ElementType, 3 > &out, AbstractFilter::Properties* properties, InterpolatorBase< Image< ElementType, 3 > >* interpolator )
{
	typename ImageTransform< ElementType, 3 >::Properties* prop = dynamic_cast< typename ImageTransform< ElementType, 3 >::Properties* >( properties );
	
	Vector< uint32, 3 > size;
	Vector< int32, 3 > strides;
	out.GetPointer( size, strides );
	//Multithreading::Thread* thr[size[2]];
	uint32 i;
	for ( i = 0; i < size[2]; ++i ) /*thr[i] = new Multithreading::Thread(*/{ TransformSlice< ElementType > ts( in, out, prop, interpolator, i ); ts();}// );
	/*for ( i = 0; i < size[2]; ++i )	thr[i]->join();
	for ( i = 0; i < size[2]; ++i )	delete thr[i];*/
	
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
void
ImageTransform< ElementType, dim >
::Rescale()
{
	int32 minimums[ dim ];
        int32 maximums[ dim ];
	float32 voxelExtents[ dim ];
	for ( uint32 d = 0; d < dim; ++d )
	{
		minimums[ d ] = this->out->GetDimensionExtents( d ).minimum;
		maximums[ d ] = this->out->GetDimensionExtents( d ).maximum;
		voxelExtents[ d ] = this->out->GetDimensionExtents( d ).elementExtent / static_cast< Properties* >( this->_properties )->_scale[ d ];
	}
	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

template< typename ElementType, uint32 dim >
bool
ImageTransform< ElementType, dim >
::ExecuteTransformation()
{
	bool result = false;
	LinearInterpolator< ImageType > interpolator( this->in );
	result = TransformImage< ElementType >( *(this->in), *(this->out), this->_properties,static_cast< InterpolatorBase< ImageType >* >( &interpolator ) );
	return result;
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
	Rescale();
	result = ExecuteTransformation();
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

	//Image of greater dimension cannot convert to image of lesser dimension
	if( this->in->GetDimension() > ImageTraits< ImageType >::Dimension ) {
		throw EDatasetTransformImpossible();
	}

	const unsigned dimension = this->in->GetDimension();
	int32 minimums[ ImageTraits<ImageType>::Dimension ];
	int32 maximums[ ImageTraits<ImageType>::Dimension ];
	float32 voxelExtents[ ImageTraits<ImageType>::Dimension ];
	Properties* prop = dynamic_cast< Properties* >( this->_properties );

	for( unsigned i=0; i < dimension; ++i ) {
		const DimensionExtents & dimExt = this->in->GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.minimum + (dimExt.maximum - dimExt.minimum) * prop->_sampling[i];
		voxelExtents[i] = dimExt.elementExtent / prop->_sampling[i];
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

