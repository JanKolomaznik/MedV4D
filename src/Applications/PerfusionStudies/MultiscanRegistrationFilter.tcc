/**
 * @author Attila Ulman
 * @file MultiscanRegistrationFilter.tcc 
 * @{ 
 **/

#ifndef MULTISCAN_REGISTRATION_FILTER_H
#error File MultiscanRegistrationFilter.tcc cannot be included directly!
#else


namespace M4D {
namespace Imaging {

/**
 * Copies input slice (inSlice) to output (outSlice). 
 *
 *  @param inSlice pointer to the input slice
 *  @param outSlice pointer to the output slice
 */
template< typename ElementType >
void CopySlice ( SliceInfo< ElementType > *inSlice, SliceInfo< ElementType > *outSlice )
{
  uint32 width  = outSlice->size[0];
  uint32 height = outSlice->size[1];

  int32 inXStride = inSlice->stride[0];
  int32 inYStride = inSlice->stride[1];
  
  int32 outXStride = outSlice->stride[0];
  int32 outYStride = outSlice->stride[1];
  
  ElementType *inPointer  = inSlice->pointer;
  ElementType *outPointer = outSlice->pointer;

	for ( uint32 j = 0; j < height; ++j ) 
  {
    ElementType *in  = inPointer  + j * inYStride;
    ElementType *out = outPointer + j * outYStride;

		for ( uint32 i = 0; i < width; ++i ) 
    {
      *out = *in;

			in  += inXStride;
      out += outXStride;
		}
	}
}


/**
 * Transforms input slice (inSlice) to output (outSlice) according to given transformation 
 * parameters (transInfo) and interpolator (interpolator). 
 *
 *  @param inSlice pointer to the input slice
 *  @param outSlice pointer to the output slice
 *  @param transInfo reference to the TransformationInfo2D structure holding actual trans. parameters
 *  @param interpolator pointer to the interpolator used during the transformation
 */
template< typename ElementType >
void TransformSlice ( SliceInfo< ElementType > *inSlice, SliceInfo< ElementType > *outSlice,
		                  TransformationInfo2D &transInfo, Interpolator2D< ElementType > *interpolator )
{
  uint32 width  = outSlice->size[0];
  uint32 height = outSlice->size[1];

  bool noSamplingWidth  = transInfo.sampling == 0 || width  < transInfo.sampling;
  bool noSamplingHeight = transInfo.sampling == 0 || height < transInfo.sampling;

  uint32 widthBorder  = (noSamplingWidth  ? width  : transInfo.sampling);
  uint32 heightBorder = (noSamplingHeight ? height : transInfo.sampling);
  
  uint32 widthMultiplicator  = (noSamplingWidth  ? 1 : width  / transInfo.sampling);
  uint32 heightMultiplicator = (noSamplingHeight ? 1 : height / transInfo.sampling);
  
  int32 xStride = outSlice->stride[0] * widthMultiplicator;
  int32 yStride = outSlice->stride[1] * heightMultiplicator;

  float32 xExtent = outSlice->extent[0];
  float32 yExtent = outSlice->extent[1];

  float32 newWidth  = width  * xExtent;
	float32 newHeight = height * yExtent;

  ElementType *pointer = outSlice->pointer;

  float32 angle = DEGTORAD(-transInfo.rotation);

  Vector< CoordType, 2 > rotationMatrix( CoordType( cos( angle ), -sin( angle ) ),
					                               CoordType( sin( angle ),  cos( angle ) ) );

  interpolator->SetParams( inSlice->pointer, inSlice->stride );

	// for each pixel, use the rotation matrix and translation to calculate the translated coordinate, and interpolate the value
	for ( uint32 j = 0; j < heightBorder; ++j ) 
  {
		ElementType *p = pointer + j * yStride;

		for ( uint32 i = 0; i < widthBorder; ++i ) 
    {
	    CoordType point( 0.0, 0.0 );

      float32 p0 = xExtent * i * widthMultiplicator  - newWidth  / 2;
      float32 p1 = yExtent * j * heightMultiplicator - newHeight / 2;

      point[0] = p0 * rotationMatrix[0][0] + p1 * rotationMatrix[0][1] + newWidth  / 2;
		  point[1] = p0 * rotationMatrix[1][0] + p1 * rotationMatrix[1][1] + newHeight / 2;
      
      point[0] /= xExtent;
			point[1] /= yExtent;

      point[0] -= transInfo.translation[0];
			point[1] -= transInfo.translation[1];

			if ( (point[0] < 0) || (point[0] > (width - 1)) ||
           (point[1] < 0) || (point[1] > (height - 1)) ) {
        *p = 0;
      } 
      else { 
        *p = interpolator->Get( point );
      }

			p += xStride;
		}
	}
}


/**
 * Calculates joint histogram of 2D image.
 *
 *  @param jointHist reference to the joint histogram's structure
 *  @param inSlice pointer to the input slice
 *  @param refSlice pointer to the reference slice
 *  @param transformSampling the sampling of transformation and histogram calculation
 */
template< typename ElementType >
void CalculateHistograms ( MultiHistogram< HistCellType, 2 > &jointHist, 
                           SliceInfo< ElementType > *inSlice, SliceInfo< ElementType > *refSlice, 
                           uint32 transformSampling )
{
	// set initial values
	jointHist.Reset();

  uint32 refWidth  = refSlice->size[0];
  uint32 refHeight = refSlice->size[1];
	
  uint32 widthBorder  = (refWidth  < transformSampling ? refWidth  : transformSampling);
  uint32 heightBorder = (refHeight < transformSampling ? refHeight : transformSampling);

  uint32 widthMultiplicator  = (refWidth  < transformSampling ? 1 : refWidth  / transformSampling);
  uint32 heightMultiplicator = (refHeight < transformSampling ? 1 : refHeight / transformSampling);

  int32 refXStride = refSlice->stride[0] * widthMultiplicator;
  int32 refYStride = refSlice->stride[1] * heightMultiplicator;
  
  int32 inXStride = inSlice->stride[0] * widthMultiplicator;
  int32 inYStride = inSlice->stride[1] * heightMultiplicator;
  
  ElementType *refPointer = refSlice->pointer;

  ElementType *inPointer = inSlice->pointer;

	vector< int32 > index( 2 );

	HistCellType base = 1.0 / (heightBorder * widthBorder);

	// loop through the transformed values and increase the histogram's value at the given position
	for ( uint32 j = 0; j < heightBorder; ++j )
	{
		ElementType *in  = inPointer  + j * inYStride;
		ElementType *ref = refPointer + j * refYStride;
		
    for ( uint32 i = 0; i < widthBorder; ++i )
		{
			// divide the values so that the histogram's size would be smaller
			index[0] = (*in)  / HISTOGRAM_VALUE_DIVISOR;
			index[1] = (*ref) / HISTOGRAM_VALUE_DIVISOR;

			jointHist.SetValueCell( index, jointHist.Get( index ) + base );

			in  += inXStride;
			ref += refXStride;
		}
	}
}


template< typename ElementType >
MultiscanRegistrationFilter< Image< ElementType, 3 > >::MultiscanRegistrationFilter () 
  : PredecessorType( new Properties() ),
    jointHistogram( vector< int32 >( 2, HISTOGRAM_MIN_VALUE ), vector< int32 >( 2, HISTOGRAM_MAX_VALUE ) )
{
  interpolator = NULL;

  inSlice = outSlice = refSlice = NULL;

  criterionComponent    = new NormalizedMutualInformation< HistCellType >();
  optimizationComponent = new PowellOptimization< MultiscanRegistrationFilter< Image< ElementType, 3 > >, double, 3 >();
}


template< typename ElementType >
MultiscanRegistrationFilter< Image< ElementType, 3 > >::MultiscanRegistrationFilter ( typename MultiscanRegistrationFilter< Image< ElementType, 3 > >::Properties *prop ) 
  : PredecessorType( prop ),
    jointHistogram( vector< int32 >( 2, HISTOGRAM_MIN_VALUE ), vector< int32 >( 2, HISTOGRAM_MAX_VALUE ) )
{
  interpolator = NULL;

  inSlice = outSlice = refSlice = NULL;

  criterionComponent    = new NormalizedMutualInformation< HistCellType >();
  optimizationComponent = new PowellOptimization< MultiscanRegistrationFilter< Image< ElementType, 3 > >, double, 3 >();
}


template< typename ElementType >
MultiscanRegistrationFilter< Image< ElementType, 3 > >::~MultiscanRegistrationFilter () 
{
  delete interpolator;

  delete inSlice;
  delete outSlice;
  delete refSlice;

  delete criterionComponent;
  delete optimizationComponent;
}


template< typename ElementType >
bool MultiscanRegistrationFilter< Image< ElementType, 3 > >::ProcessImage ( const Image< ElementType, 3 > &in,
			                                                                            Image< ElementType, 3 >	&out )
{
	switch ( GetInterpolationType() ) 
  {
	  case IT_NEAREST:
		  interpolator = new NearestInterpolator2D< ElementType >();
      break;
	  
    case IT_LINEAR:
		  interpolator = new LinearInterpolator2D< ElementType >();
      break;
	  
    default:
		  ASSERT(false);
	}

  return ProcessImageHelper( in, out );
}


template< typename ElementType >
bool MultiscanRegistrationFilter< Image< ElementType, 3 > >::ProcessImageHelper ( const Image< ElementType, 3 > &in,
                                                                                    	  Image< ElementType, 3 > &out )
{
  Vector< uint32, 3 > iSize;
	Vector< int32, 3 > iStrides;
	ElementType *inPointer = in.GetPointer( iSize, iStrides );

	Vector< uint32, 3 > oSize;
	Vector< int32, 3 > oStrides;
	ElementType *outPointer = out.GetPointer( oSize, oStrides );

  inSlice  = new SliceInfo< ElementType >( inPointer,  Vector< uint32, 2 >( iSize[0], iSize[1] ), 
                                           Vector< int32, 2 >( iStrides[0], iStrides[1] ),
                                           Vector< float32, 2 >( in.GetDimensionExtents( 0 ).elementExtent, 
                                                                 in.GetDimensionExtents( 1 ).elementExtent ) );
  outSlice = new SliceInfo< ElementType >( outPointer, Vector< uint32, 2 >( oSize[0], oSize[1] ),
                                           Vector< int32, 2 >( oStrides[0], oStrides[1] ),
                                           Vector< float32, 2 >( out.GetDimensionExtents( 0 ).elementExtent, 
                                                                 out.GetDimensionExtents( 1 ).elementExtent ) );

  refSlice = new SliceInfo< ElementType >();

	int32	 izStride = iStrides[2];
  int32	 ozStride = oStrides[2];

	uint32 depth    = oSize[2];
  uint32 sliceNum = GetExaminedSliceNum();
  uint32 times    = (uint32)(depth / sliceNum);

  if ( depth % sliceNum ) {
    return false;
  }

  for ( uint32 i = 0; i < sliceNum; ++i )
  {
    inSlice->pointer = inPointer + i * izStride;
    *refSlice = *inSlice;

    // first slice is reference - just copy it to output
    CopySlice( inSlice, outSlice );
    

    inSlice->pointer  += sliceNum * izStride;
		outSlice->pointer += ozStride;

    for ( uint32 j = 1; j < times; ++j ) 
    {
      // register slices - slices in 1 time sequence 
      // (1st one is the reference slice, next slices in the time sequence are transformed according to the first one)
      
      if ( GetRegistrationNeeded() ) 
      {
        if ( !RegisterSlice() ) {
          return false;
        }  
      }
      else {
        CopySlice( inSlice, outSlice );
      }
      
  	  inSlice->pointer  += sliceNum * izStride;
		  outSlice->pointer += ozStride;
	  }
  }

  return true;
}


template< typename ElementType >
bool MultiscanRegistrationFilter< Image< ElementType, 3 > >::RegisterSlice ()
{
  transformationInfo.Reset();

	uint32 maxSampling = transformationInfo.sampling;

  for ( transformationInfo.sampling = MIN_SAMPLING; transformationInfo.sampling <= maxSampling && this->IsRunning(); transformationInfo.sampling *= 2 )
  {
	  LOG( "Resolution: " << transformationInfo.sampling );
	  
    Vector< double, 3 > v;
	  for ( uint32 i = 0; i < 2; ++i ) {
		  v[i] = transformationInfo.translation[i];
	  }
    v[2] = transformationInfo.rotation;

	  double fret;
	  optimizationComponent->optimize( v, fret, this );
  }

  transformationInfo.sampling = 0;

  if ( !this->IsRunning() ) {
    return false;
  }

  // actual transformation
  TransformSlice( inSlice, outSlice, transformationInfo, interpolator );

  return true;
}


template< typename ElementType >
double MultiscanRegistrationFilter< Image< ElementType, 3 > >::OptimizationFunction ( Vector< double, 3 > &v )
{
  // reset parameters
  CoordType trans;
  for ( uint32 i = 0; i < 2; ++i ) {
		trans[i] = v[i];
	}

  float32 rot = v[2];

	// reset rotation and translation
	transformationInfo.SetParams( trans, rot );

  LOG( "Current registration parameters - rotation:  " << transformationInfo.rotation << ", "
                                   << "translation:  " << transformationInfo.translation );

	double res = 1.0;

	if ( this->IsRunning() )
	{
		// transform image and calculate criterion
    TransformSlice( inSlice, outSlice, transformationInfo, interpolator );

		CalculateHistograms< ElementType >( jointHistogram, outSlice, refSlice, transformationInfo.sampling );
		
    res = criterionComponent->compute( jointHistogram );
		
    LOG( "Mutual Information value: " << res );
	}

	return 1.0 / res;
}

} // namespace Imaging
} // namespace M4D


#endif // MULTISCAN_REGISTRATION_FILTER_H

/** @} */

