/**
 * @author Attila Ulman
 * @file MultiscanSegmentationFilter.tcc 
 * @{ 
 **/

#ifndef MULTISCAN_SEGMENTATION_FILTER_H
#error File MultiscanSegmentationFilter.tcc cannot be included directly!
#else


namespace M4D {
namespace Imaging {

/**
 * Calculates homogenity criterion - if it's possible to fill the next pixel.
 *
 *  @param pointer pointer to the input slice
 *  @param mask pointer to the binary mask which is about to be created
 *  @param offset offset determining the position within the input data and mask
 *  @param boneDensityBottom value for the homogenity criterion - lower bound of the "bone interval"
 *  @return true if the flood can proceed (criterion holds & not already in mask), false otherwise
 */
template< typename ElementType >
bool canFill ( ElementType *pointer, bool *mask, int32 offset, ElementType boneDensityBottom )
{
  if ( *(pointer + offset) < boneDensityBottom && !*(mask + offset) ) {
    return true;
  } else {
    return false;
  }
}


/**
 * Creates binary mask (parameter mask) - segmented brain shape.
 * Region growing algorithm - scanline floodfill with stack.
 *
 *  @param slice pointer to the input slice
 *  @param mask pointer to the preallocated space for the mask to be created
 *  @param boneDensityBottom value for the homogenity criterion - lower bound of the "bone interval"
 */
template< typename ElementType >
void ComputeMask ( SliceInfo< ElementType > *slice, bool *mask, ElementType	boneDensityBottom )
{
  uint32 width  = slice->size[0];
  uint32 height = slice->size[1];

  int32 xStride = slice->stride[0];
  int32 yStride = slice->stride[1];
  
  ElementType *pointer  = slice->pointer;

	for ( uint32 j = 0; j < height; ++j )
  {
    bool *m = mask + j * yStride;

		for ( uint32 i = 0; i < width; ++i ) 
    {
      *m = false; 

      m += xStride;
    }
	}

  stack< CoordType > stack;
  stack.push( CoordType( width / 2, height / 2 ) );

  uint32 x, y;
  bool spanLeft, spanRight;

  while ( !stack.empty() )
  {    
    CoordType coord = stack.top();
    stack.pop();

    x = (uint32)coord[0]; 
    y = (uint32)coord[1];

    spanLeft = spanRight = FALSE;

    while ( y > 0 && canFill( pointer, mask, y * yStride + x * xStride, boneDensityBottom ) ) {
      y--;
    }
    y++;

    int32 offset = x * xStride + y * yStride;
    
    while ( y < height && canFill( pointer, mask, offset, boneDensityBottom ) )
    { 
      *(mask + offset) = true;

      if ( !spanLeft && x > 0 && canFill( pointer, mask, offset - xStride, boneDensityBottom ) ) 
      {
        stack.push( CoordType( x - 1, y ) );
        spanLeft = 1;
      }
      else if ( spanLeft && x > 0 && !canFill( pointer, mask, offset - xStride, boneDensityBottom ) ) {
        spanLeft = 0;
      }

      if ( !spanRight && x < (width - 1) && canFill( pointer, mask, offset + xStride, boneDensityBottom ) ) 
      {
        stack.push( CoordType( x + 1, y ) );
        spanRight = 1;
      }
      else if ( spanRight && x < (width - 1) && !canFill( pointer, mask, offset + xStride, boneDensityBottom ) ) {
        spanRight = 0;
      } 

      y++;
      offset += yStride;
    }
  }
}


/**
 * Masks input slice (inSlice) to output (outSlice) by using binary mask (mask) - segmented brain shape.
 *
 *  @param inSlice pointer to the input slice
 *  @param outSlice pointer to the output slice
 *  @param mask pointer to the binary mask used for masking
 *  @param background value of the background in the result 
 */
template< typename ElementType >
void Mask ( SliceInfo< ElementType > *inSlice, SliceInfo< ElementType > *outSlice, 
            bool *mask, ElementType	background )
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

    bool *m = mask + j * outYStride;

		for ( uint32 i = 0; i < width; ++i ) 
    {
      if ( *m ) {
		    *out = *in;
	    } else {
		    *out = background;
	    }

			in  += inXStride;
      out += outXStride;

      m += outXStride;
		}
	}
}


template< typename ElementType >
MultiscanSegmentationFilter< Image< ElementType, 3 > >::MultiscanSegmentationFilter () 
  : PredecessorType( new Properties() )
{}


template< typename ElementType >
MultiscanSegmentationFilter< Image< ElementType, 3 > >::MultiscanSegmentationFilter ( typename MultiscanSegmentationFilter< Image< ElementType, 3 > >::Properties *prop ) 
  : PredecessorType( prop )
{}


template< typename ElementType >
MultiscanSegmentationFilter< Image< ElementType, 3 > >::~MultiscanSegmentationFilter () 
{}


template< typename ElementType >
bool MultiscanSegmentationFilter< Image< ElementType, 3 > >::ProcessImage ( const Image< ElementType, 3 > &in,
			                                                                      Image< ElementType, 3 >	&out )
{
  Vector< uint32, 3 > iSize;
	Vector< int32, 3 > iStrides;
	ElementType *inPointer = in.GetPointer( iSize, iStrides );

	Vector< uint32, 3 > oSize;
	Vector< int32, 3 > oStrides;
	ElementType *outPointer = out.GetPointer( oSize, oStrides );

  SliceInfo< ElementType > *inSlice, *outSlice;
  inSlice  = new SliceInfo< ElementType >( inPointer,  Vector< uint32, 2 >( iSize[0], iSize[1] ), 
                                           Vector< int32, 2 >( iStrides[0], iStrides[1] ),
                                           Vector< float32, 2 >( in.GetDimensionExtents( 0 ).elementExtent, 
                                                                 in.GetDimensionExtents( 1 ).elementExtent ) );
  outSlice = new SliceInfo< ElementType >( outPointer, Vector< uint32, 2 >( oSize[0], oSize[1] ),
                                           Vector< int32, 2 >( oStrides[0], oStrides[1] ),
                                           Vector< float32, 2 >( out.GetDimensionExtents( 0 ).elementExtent, 
                                                                 out.GetDimensionExtents( 1 ).elementExtent ) );
  
  bool *mask = new bool[ oSize[1] * oStrides[1] ];

	int32	 izStride = iStrides[2];
  int32	 ozStride = oStrides[2];

	uint32 depth    = oSize[2];
  uint32 sliceNum = GetExaminedSliceNum();
  uint32 times    = (uint32)(depth / sliceNum);

  ElementType	boneDensityBottom = GetBoneDensityBottom();

  ElementType	background = GetBackground();

  for ( uint32 i = 0; i < sliceNum; ++i )
  {
    // first slice - compute mask
    ComputeMask( inSlice, mask, boneDensityBottom );

    for ( uint32 j = 0; j < times; ++j ) 
    {
      // segmentate
      Mask( inSlice, outSlice, mask, background );

		  inSlice->pointer  += izStride;
		  outSlice->pointer += ozStride;
	  }
  }

  // cleanup
  delete inSlice;
  inSlice = NULL;

  delete outSlice;
  outSlice = NULL;

  delete [] mask; 
  mask = NULL;

  return true;
}

} // namespace Imaging
} // namespace M4D


#endif // MULTISCAN_SEGMENTATION_FILTER_H

/** @} */

