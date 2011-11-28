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

/**
 * Transform a 2D image
 *  @param out reference to the output image
 *  @param properties pointer to a properties structure
 *  @param transformSampling the subsampling to use for transformation (irrelevant in this case)
 *  @param threadNumber the number of parallel slice-computing threads to use at once (irrelevant in this case)
 *  @return true on success, false otherwise
 */
template< typename ElementType >
bool
TransformImage( Image< ElementType, 2 > &out,
		AFilter::Properties* properties,
		InterpolatorBase< Image< ElementType, 2 > >* interpolator,
		uint32 transformSampling,
		uint32 threadNumber )
{
	
	ElementType *sPointer;
	int32 xStride;
	int32 yStride;
	int32 height, oldheight;
	int32 width, oldwidth;
	float32 xExtent, yExtent;
	
	Vector< uint32, 2 > size;
	Vector< int32, 2 > strides;

	typename ImageTransform< ElementType, 2 >::Properties* prop = dynamic_cast< typename ImageTransform< ElementType, 2 >::Properties* >( properties );
	typedef typename InterpolatorBase< Image< ElementType, 2 > >::CoordType CoordType;
	sPointer = out.GetPointer( size, strides );
	width = size[0];
	height = size[1];
	xStride = strides[0];
	yStride = strides[1];
	oldwidth = (int32)(width / prop->_sampling[0]);
	oldheight = (int32)(height / prop->_sampling[1]);
	xExtent = out.GetDimensionExtents( 0 ).elementExtent;
	yExtent = out.GetDimensionExtents( 1 ).elementExtent;

	Vector< CoordType, 2 > RotationMatrix(
					CoordType( std::cos( -prop->_rotation[0] ), -std::sin( -prop->_rotation[0] ) ),
					CoordType( std::sin( -prop->_rotation[0] ),  std::cos( -prop->_rotation[0] ) )
					);

	// for each pixel, use the rotation matrix, translation and scale
	// to calculate the translated coordinate, and interpolate the value
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

/**
 * Class for threads to do the transformation of each slice
 */
template< typename ElementType >
class TransformSlice
{
public:

	/**
	 * Constructor
	 *  @param output reference to the output image
	 *  @param properties pointer to a properties structure
	 *  @param interp pointer to the interpolator to use
	 *  @param sNum the slice number
	 *  @param tSampling the subsampling to use for transformation
	 *  @param rotMatrix the rotation matrix to use for transformation
	 */
	TransformSlice( Image< ElementType, 3 > &output,
			typename ImageTransform< ElementType, 3 >::Properties* properties,
			InterpolatorBase< Image< ElementType, 3 > >* interp,
			int32 sNum,
			uint32 tSampling,
			Vector< typename InterpolatorBase< Image< ElementType, 3 > >::CoordType, 3 >& rotMatrix )
		: out( output ),
		  prop( properties ),
		  interpolator( interp ),
		  sliceNum( sNum ),
		  transformSampling( tSampling ),
		  RotationMatrix( rotMatrix )
	{

		// initialize parameters
		sPointer = out.GetPointer( size, strides );
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

		
		scaleBias = Vector< int32, 3 >(
			( width / 2 - width / ( 2 * prop->_scale[0] ) ),
			( height / 2 - height / ( 2 * prop->_scale[1] ) ),
			( depth / 2 - depth / ( 2 * prop->_scale[2] ) )
		);

		heightBorder = (int32)( ( transformSampling == 0 || height < (int32)transformSampling ) ? height : transformSampling );
		widthBorder = (int32)( ( transformSampling == 0 || width < (int32)transformSampling ) ? width : transformSampling );
		heightMultiplicator = ( ( transformSampling == 0 || height < (int32)transformSampling ) ? 1 : height / transformSampling );
		widthMultiplicator = ( ( transformSampling == 0 || width < (int32)transformSampling ) ? 1 : width / transformSampling );
	}

	void operator()()
	{
		typedef typename InterpolatorBase< Image< ElementType, 3 > >::CoordType CoordType;

		int32 k = sliceNum;
		if ( k >= depth ) return;

		// initialize starting points by transforming them
		CoordType startpoint = calculatePoint( 0.0, 0.0, k );
		CoordType xdiff = calculatePoint( 1.0, 0.0, k );
		CoordType ydiff = calculatePoint( 0.0, 1.0, k );

		// calculate differences
		xdiff[0] -= startpoint[0];
		xdiff[1] -= startpoint[1];
		xdiff[2] -= startpoint[2];

		ydiff[0] -= startpoint[0];
		ydiff[1] -= startpoint[1];
		ydiff[2] -= startpoint[2];

		CoordType point( 0.0, 0.0, 0.0 );

		CoordType rowstartpoint = startpoint;

		int32 stridedHeightMultiplicator = yStride * heightMultiplicator;

		int32 stridedWidthMultiplicator = xStride * widthMultiplicator;

		// for each pixel on the slice, get the transformation using
		// the starting point and the x and y differences, and interpolate the value
		for( int32 j = 0; j < heightBorder; ++j ) {

			pointer = sPointer + k*zStride + j * stridedHeightMultiplicator;

			point = rowstartpoint;

			for( int32 i = 0; i < widthBorder; ++i ) {

				if ( ( point[0] < 0 ) |
				     ( point[0] > ( oldwidth - 1 ) ) |
				     ( point[1] < 0 ) |
			   	     ( point[1] > ( oldheight - 1 ) ) |
			   	     ( point[2] < 0 ) |
			   	     ( point[2] > ( olddepth - 1 ) ) ) *pointer = 0;

				else *pointer = interpolator->Get( point );
			
				pointer += stridedWidthMultiplicator;

				point += xdiff;
			}

			rowstartpoint += ydiff;
		}
	}
private:

	/**
	 * Calculate the transformation of a single coordinate
	 *  @param xcoord the x coordinate
	 *  @param ycoord the y coordinate
	 *  @param zcoord the z coordinate
	 *  @return the interpolated coordinates
	 */
	typename InterpolatorBase< Image< ElementType, 3 > >::CoordType calculatePoint( int32 xcoord, int32 ycoord, int32 zcoord )
	{
		typedef typename InterpolatorBase< Image< ElementType, 3 > >::CoordType CoordType;

		CoordType point( 0.0, 0.0, 0.0 );

		// rotate the coordinates
		point[0] = ( xExtent * xcoord * widthMultiplicator - newwidth/2 ) * RotationMatrix[0][0] + ( yExtent * ycoord * heightMultiplicator - newheight/2 ) * RotationMatrix[0][1] + ( zExtent * zcoord - newdepth/2 ) * RotationMatrix[0][2] + newwidth/2;
		point[1] = ( xExtent * xcoord * widthMultiplicator - newwidth/2 ) * RotationMatrix[1][0] + ( yExtent * ycoord * heightMultiplicator - newheight/2 ) * RotationMatrix[1][1] + ( zExtent * zcoord - newdepth/2 ) * RotationMatrix[1][2] + newheight/2;
		point[2] = ( xExtent * xcoord * widthMultiplicator - newwidth/2 ) * RotationMatrix[2][0] + ( yExtent * ycoord * heightMultiplicator - newheight/2 ) * RotationMatrix[2][1] + ( zExtent * zcoord - newdepth/2 ) * RotationMatrix[2][2] + newdepth/2;

		// resize according to extents
		point[0] /= xExtent;
		point[1] /= yExtent;
		point[2] /= zExtent;

		// translate the point
		point[0] -= prop->_translation[0];
		point[1] -= prop->_translation[1];
		point[2] -= prop->_translation[2];

		// rescale the point
		point[0] = ( point[0] / prop->_scale[0] + scaleBias[0] );
		point[1] = ( point[1] / prop->_scale[1] + scaleBias[1] );
		point[2] = ( point[2] / prop->_scale[2] + scaleBias[2] );

		// resample the point
		point[0] /= prop->_sampling[0];
		point[1] /= prop->_sampling[1];
		point[2] /= prop->_sampling[2];

		return point;

	}

	// output image
	Image< ElementType, 3 > &out;

	// properties structure
	typename ImageTransform< ElementType, 3 >::Properties* prop;

	// interpolator
	InterpolatorBase< Image< ElementType, 3 > >* interpolator;

	// slice number
	int32 sliceNum;

	// transform sampling
	uint32 transformSampling;

	// rotation matrix
	Vector< typename InterpolatorBase< Image< ElementType, 3 > >::CoordType, 3 >& RotationMatrix;

	// pointers to positions in the value array of the dataset
	ElementType *sPointer, *pointer;

	//strides
	int32 xStride;
	int32 yStride;
	int32 zStride;

	// height and height-connected values
	int32 height, oldheight, heightBorder, heightMultiplicator;

	// width and width-connected values
	int32 width, oldwidth, widthBorder, widthMultiplicator;

	// depth
	int32 depth, olddepth;

	// extents
	float32 xExtent, yExtent, zExtent;

	// recalculated values for width, height, depth
	float32 newwidth, newheight, newdepth;

	// sizes, strides and scale biases
	Vector< uint32, 3 > size;
	Vector< int32, 3 > strides;
	Vector< int32, 3 > scaleBias;
};

/**
 * Transform a 3D image
 *  @param out reference to the output image
 *  @param properties pointer to a properties structure
 *  @param transformSampling the subsampling to use for transformation
 *  @param threadNumber the number of parallel slice-computing threads to use at once
 *  @return true on success, false otherwise
 */
template< typename ElementType >
bool
TransformImage(  Image< ElementType, 3 > &out,
		AFilter::Properties* properties,
		InterpolatorBase< Image< ElementType, 3 > >* interpolator,
		uint32 transformSampling,
		uint32 threadNumber )
{
	typename ImageTransform< ElementType, 3 >::Properties* prop = dynamic_cast< typename ImageTransform< ElementType, 3 >::Properties* >( properties );

	typedef typename InterpolatorBase< Image< ElementType, 3 > >::CoordType CoordType;
	
	Vector< uint32, 3 > size;
	Vector< int32, 3 > strides;
	out.GetPointer( size, strides );

	int32 depth = size[2];

	// rotation matrices	
	Vector< CoordType, 3 > RotationMatrixX(
					CoordType( 1.0, 0.0, 0.0 ),
					CoordType( 0.0, std::cos( -prop->_rotation[0] ), -std::sin( -prop->_rotation[0] ) ),
					CoordType( 0.0, std::sin( -prop->_rotation[0] ),  std::cos( -prop->_rotation[0] ) )
					);

	Vector< CoordType, 3 > RotationMatrixY(
					CoordType( std::cos( -prop->_rotation[1] ), 0.0,  std::sin( -prop->_rotation[1] ) ),
					CoordType( 0.0, 1.0, 0.0 ),
					CoordType( -std::sin( -prop->_rotation[1] ), 0.0, std::cos( -prop->_rotation[1] ) )
					);

	Vector< CoordType, 3 > RotationMatrixZ(
					CoordType( std::cos( -prop->_rotation[2] ), -std::sin( -prop->_rotation[2] ), 0.0 ),
					CoordType( std::sin( -prop->_rotation[2] ),  std::cos( -prop->_rotation[2] ), 0.0 ),
					CoordType( 0.0, 0.0, 1.0 )
					);

	// multiply rotation matrices
	Vector< CoordType, 3 > RotationMatrix(
					CoordType( RotationMatrixX[0][0] * RotationMatrixY[0][0] + RotationMatrixX[0][1] * RotationMatrixY[1][0] + RotationMatrixX[0][2] * RotationMatrixY[2][0],
						   RotationMatrixX[0][0] * RotationMatrixY[0][1] + RotationMatrixX[0][1] * RotationMatrixY[1][1] + RotationMatrixX[0][2] * RotationMatrixY[2][1],
						   RotationMatrixX[0][0] * RotationMatrixY[0][2] + RotationMatrixX[0][1] * RotationMatrixY[1][2] + RotationMatrixX[0][2] * RotationMatrixY[2][2] ),
					CoordType( RotationMatrixX[1][0] * RotationMatrixY[0][0] + RotationMatrixX[1][1] * RotationMatrixY[1][0] + RotationMatrixX[1][2] * RotationMatrixY[2][0],
                                                   RotationMatrixX[1][0] * RotationMatrixY[0][1] + RotationMatrixX[1][1] * RotationMatrixY[1][1] + RotationMatrixX[1][2] * RotationMatrixY[2][1],
                                                   RotationMatrixX[1][0] * RotationMatrixY[0][2] + RotationMatrixX[1][1] * RotationMatrixY[1][2] + RotationMatrixX[1][2] * RotationMatrixY[2][2] ),
					CoordType( RotationMatrixX[2][0] * RotationMatrixY[0][0] + RotationMatrixX[2][1] * RotationMatrixY[1][0] + RotationMatrixX[2][2] * RotationMatrixY[2][0],
                                                   RotationMatrixX[2][0] * RotationMatrixY[0][1] + RotationMatrixX[2][1] * RotationMatrixY[1][1] + RotationMatrixX[2][2] * RotationMatrixY[2][1],
                                                   RotationMatrixX[2][0] * RotationMatrixY[0][2] + RotationMatrixX[2][1] * RotationMatrixY[1][2] + RotationMatrixX[2][2] * RotationMatrixY[2][2] )
					);

	RotationMatrix = Vector< CoordType, 3 >(
					CoordType( RotationMatrix[0][0] * RotationMatrixZ[0][0] + RotationMatrix[0][1] * RotationMatrixZ[1][0] + RotationMatrix[0][2] * RotationMatrixZ[2][0],
						   RotationMatrix[0][0] * RotationMatrixZ[0][1] + RotationMatrix[0][1] * RotationMatrixZ[1][1] + RotationMatrix[0][2] * RotationMatrixZ[2][1],
						   RotationMatrix[0][0] * RotationMatrixZ[0][2] + RotationMatrix[0][1] * RotationMatrixZ[1][2] + RotationMatrix[0][2] * RotationMatrixZ[2][2] ),
					CoordType( RotationMatrix[1][0] * RotationMatrixZ[0][0] + RotationMatrix[1][1] * RotationMatrixZ[1][0] + RotationMatrix[1][2] * RotationMatrixZ[2][0],
                                                   RotationMatrix[1][0] * RotationMatrixZ[0][1] + RotationMatrix[1][1] * RotationMatrixZ[1][1] + RotationMatrix[1][2] * RotationMatrixZ[2][1],
                                                   RotationMatrix[1][0] * RotationMatrixZ[0][2] + RotationMatrix[1][1] * RotationMatrixZ[1][2] + RotationMatrix[1][2] * RotationMatrixZ[2][2] ),
					CoordType( RotationMatrix[2][0] * RotationMatrixZ[0][0] + RotationMatrix[2][1] * RotationMatrixZ[1][0] + RotationMatrix[2][2] * RotationMatrixZ[2][0],
                                                   RotationMatrix[2][0] * RotationMatrixZ[0][1] + RotationMatrix[2][1] * RotationMatrixZ[1][1] + RotationMatrix[2][2] * RotationMatrixZ[2][1],
                                                   RotationMatrix[2][0] * RotationMatrixZ[0][2] + RotationMatrix[2][1] * RotationMatrixZ[1][2] + RotationMatrix[2][2] * RotationMatrixZ[2][2] )
					);


	uint32 thread_num = (threadNumber < size[2]) ? (threadNumber > 1 ? threadNumber : 1) : size[2];
	Multithreading::Thread** thr = new Multithreading::Thread*[ thread_num ];
	uint32 i = 0, j;

	int32 depthBorder = (int32)( ( transformSampling == 0 || depth < (int32)transformSampling ) ? depth : transformSampling );
	int32 depthMultiplicator = ( ( transformSampling == 0 || depth < (int32)transformSampling ) ? 1 : depth / transformSampling );

	// execute transformation of slices one by one
	for ( i = 0, j = 0; i < thread_num && j < (uint32)depthBorder; i++, j++ ) thr[i] = new Multithreading::Thread( TransformSlice< ElementType >( out, prop, interpolator, j * depthMultiplicator, transformSampling, RotationMatrix ) );

	if ( i < thread_num )
	{
		for ( i = 0; i < j; ++i )
		{
			thr[i]->join();
			delete thr[i];
		}
		return true;
	}

	// keep in mind the maximum number of thread and the number of slices while distributing the transformation jobs among threads
	while ( j < (uint32)depthBorder )
	{
		i = i % thread_num;
		thr[i]->join();
		delete thr[i];
		thr[i] = new Multithreading::Thread( TransformSlice< ElementType >( out, prop, interpolator, j * depthMultiplicator, transformSampling, RotationMatrix ) );
		i++;
		j++;
	}

	for ( i = 0; i < thread_num; ++i )	thr[i]->join();
	for ( i = 0; i < thread_num; ++i )	delete thr[i];
	
	delete[] thr;

	return true;
}

//***************************************************************************************************************

template< typename ElementType, uint32 dim >
ImageTransform< ElementType, dim >
::ImageTransform( typename ImageTransform< ElementType, dim >::Properties  * prop )
	: PredecessorType( prop ),
	  _threadNumber( 1 ),
	  _interpolator( NULL )
{
	this->_name = "ImageTransform";
}

template< typename ElementType, uint32 dim >
ImageTransform< ElementType, dim >
::ImageTransform()
	: PredecessorType( new Properties() ),
	  _threadNumber( 1 ),
	  _interpolator( NULL )
	
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

	// reset voxel extents according to scale
	for ( uint32 d = 0; d < dim; ++d )
	{
		minimums[ d ] = this->out->GetDimensionExtents( d ).minimum;
		maximums[ d ] = this->out->GetDimensionExtents( d ).maximum;
		voxelExtents[ d ] = this->out->GetDimensionExtents( d ).elementExtent / static_cast< Properties* >( this->_properties )->_scale[ d ];
	}
	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

template< typename ElementType, uint32 dim >
void
ImageTransform< ElementType, dim >
::SetInterpolator( InterpolatorBase< ImageType >* interpolator )
{
	_interpolator = interpolator;
}

template< typename ElementType, uint32 dim >
bool
ImageTransform< ElementType, dim >
::ExecuteTransformation( uint32 transformSampling )
{
	bool result = false;
	if ( _interpolator == NULL )
	{
		_THROW_ ENULLPointer( "Interpolator not set in ImageTransform!" );
	}

	// set interpolator input image and execute transformation according to dimension
	_interpolator->SetImage( static_cast< const ImageType* >( this->in ) );
	result = TransformImage( *(this->out), this->_properties, _interpolator, transformSampling, _threadNumber );
	return result;
}

template< typename ElementType, uint32 dim >
bool
ImageTransform< ElementType, dim >
::ExecutionThreadMethod( APipeFilter::UPDATE_TYPE utype )
{
	utype = utype;
	if ( !( _readerBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
		_writerBBox->SetState( MS_CANCELED );
		return false;
	}
	bool result = false;

	// rescale and execute transformation
	Rescale();
	result = ExecuteTransformation( 0 );
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
::SetThreadNumber( uint32 tNumber )
{
	_threadNumber = tNumber;
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

	// set dimension minimums, maximums and voxel extents
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
::BeforeComputation( APipeFilter::UPDATE_TYPE &utype )
{

	PredecessorType::BeforeComputation( utype );

	//This kind of filter computes always on whole dataset
	utype = APipeFilter::RECALCULATION;
}

template< typename ElementType, uint32 dim >
void
ImageTransform< ElementType, dim >
::MarkChanges( APipeFilter::UPDATE_TYPE utype )
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

