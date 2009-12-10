/**
 * @ingroup imaging 
 * @author Attila Ulman
 * @file MultiscanRegistration.tcc 
 * @{ 
 **/

#ifndef MULTISCAN_REGISTRATION_H
#error File MultiscanRegistration.tcc cannot be included directly!
#else


/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D {
namespace Imaging {

template< typename ElementType >
MultiscanRegistration< Image< ElementType, 3 > >::MultiscanRegistration () 
  : PredecessorType( new Properties() )
{}


template< typename ElementType >
MultiscanRegistration< Image< ElementType, 3 > >::MultiscanRegistration ( typename MultiscanRegistration< Image< ElementType, 3 > >::Properties *prop ) 
  : PredecessorType( prop ) 
{}


template< typename ElementType >
bool MultiscanRegistration< Image< ElementType, 3 > >::ProcessImage ( const Image< ElementType, 3 > &in,
			                                                                      Image< ElementType, 3 >	&out )
{
	switch ( GetInterpolationType() ) 
  {
	  case IT_NEAREST:
		  return ProcessImageHelper< NearestInterpolator2D< ElementType > >( in, out );
	  
    case IT_LINEAR:
		  return ProcessImageHelper< LinearInterpolator2D< ElementType > >( in, out );
	  
    default:
		  ASSERT(false);
	}
}

template< typename ElementType >
template< typename InterpolationType >
bool MultiscanRegistration< Image< ElementType, 3 > >::ProcessImageHelper ( const Image< ElementType, 3 > &in,
                                                                               	  Image< ElementType, 3 > &out )
{
  Vector< uint32, 3 > iSize;
	Vector< int32, 3 > iStrides;
	ElementType *inPointer = in.GetPointer( iSize, iStrides );

	Vector< uint32, 3 > oSize;
	Vector< int32, 3 > oStrides;
	ElementType *outPointer = out.GetPointer( oSize, oStrides );

  SliceInfo< ElementType >  inSlice( inPointer,  Vector< uint32, 2 >( iSize[0], iSize[1] ), 
                                     Vector< int32, 2 >( iStrides[0], iStrides[1] ),
                                     Vector< float32, 2 >( in.GetDimensionExtents( 0 ).elementExtent, 
                                                           in.GetDimensionExtents( 1 ).elementExtent ) );
  SliceInfo< ElementType > outSlice( outPointer, Vector< uint32, 2 >( oSize[0], oSize[1] ),
                                     Vector< int32, 2 >( oStrides[0], oStrides[1] ),
                                     Vector< float32, 2 >( out.GetDimensionExtents( 0 ).elementExtent, 
                                                           out.GetDimensionExtents( 1 ).elementExtent ) );

	int32	 izStride = iStrides[2];
  int32	 ozStride = oStrides[2];

	uint32 depth    = oSize[2];
  uint32 sliceNum = GetExaminedSliceNum();
  uint32 times    = (uint32)(depth / sliceNum);

  if ( depth % sliceNum ) {
    // TODO throw exception
  }

  for ( uint32 i = 0; i < sliceNum; ++i )
  {
    inSlice.pointer = inPointer + i * izStride;

    for ( uint32 j = 0; j < times; ++j ) 
    {
      if ( !TransformSlice< InterpolationType >( inSlice, outSlice, TransformationInfo2D() ) ) {
        return false;
      }

		  inSlice.pointer  += sliceNum * izStride;
		  outSlice.pointer += ozStride;
	  }
  }

  return true;
}


template< typename ElementType >
template< typename InterpolationType >
bool MultiscanRegistration< Image< ElementType, 3 > >::TransformSlice ( SliceInfo< ElementType > &inSlice, 
                                                                        SliceInfo< ElementType > &outSlice,
		                                                                    TransformationInfo2D &transInfo )
{
  int32 width  = outSlice.size[0];
  int32 height = outSlice.size[1];
  
  int32 xStride = outSlice.stride[0];
  int32 yStride = outSlice.stride[1];
  
  float32 xExtent = outSlice.extent[0];
  float32 yExtent = outSlice.extent[1];

  ElementType *pointer = outSlice.pointer;

	Vector< CoordType, 2 > rotationMatrix ( CoordType( std::cos( -transInfo.rotation[0] ), -std::sin( -transInfo.rotation[0] ) ),
					                                CoordType( std::sin( -transInfo.rotation[0] ),  std::cos( -transInfo.rotation[0] ) ) );

  int32 oldWidth  = (int32)( width  / transInfo.sampling[0] );
	int32 oldHeight = (int32)( height / transInfo.sampling[1] );

  InterpolationType interpolator( inSlice.pointer, inSlice.stride );

	// for each pixel, use the rotation matrix and translation to calculate the translated coordinate, and interpolate the value
	for ( int32 j = 0; j < height; ++j ) 
  {
		ElementType *p = pointer + j * yStride;

		for ( int32 i = 0; i < width; ++i ) 
    {
			CoordType point( ((i - width / 2) * rotationMatrix[0][0] * xExtent + (j - height / 2) * rotationMatrix[0][1] * yExtent) / xExtent + width / 2,
					             ((i - width / 2) * rotationMatrix[1][0] * xExtent + (j - height / 2) * rotationMatrix[1][1] * yExtent) / yExtent + height / 2 );

			point[0] -= transInfo.translation[0];
			point[1] -= transInfo.translation[1];

			point[0] /= transInfo.sampling[0];
			point[1] /= transInfo.sampling[1];

			if ( (point[0] < 0) || (point[0] > (oldWidth - 1)) ||
           (point[1] < 0) || (point[1] > (oldHeight - 1)) ) {
        *p = 0;
      } 
      else { 
        *p = interpolator.Get( point );
      }

			p += xStride;
		}
	}

	return true;
}

} // namespace Imaging
} // namespace M4D

/** @} */


#endif // MULTISCAN_REGISTRATION_H

/** @} */

