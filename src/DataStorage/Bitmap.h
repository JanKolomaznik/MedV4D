#ifndef BITMAP_H
#define BITMAP_H

namespace Bmp {


/**
 * Arbitrary-dimension bitmaps with arbitrary pixels
 *
 * FIXME: Is Dimensions really needed? I'd rather pass arguments as numbers
 *  instead of varible-sized arrays.
 *
 */

template <int Dimensions, class PixelType>
class Bitmap {

// The bitmap (contents can be changed by using the [] operator)
  PixelType const *bmp;

public:

 // Empty bitmap: allocates the space
  Bitmap(int* widths);                           // assume low=0; strides are linear
  Bitmap(int* lows, int* widths, int* strides);  // customized

 // Initialization from an existing PixelType*
  Bitmap(PixelType* bmp, int* widths);
  Bitmap(PixelType* bmp, int* lows, int* widths, int* strides);

 // TODO: constructors with number of parameters dependent on Dimensions

 // Initialization from another Bitmap of the same type
  Bitmap(const Bmp::Bitmap<Dimensions, PixelType> &);

 // TODO: Shallow copy: points into an exising bitmap, can do 1-to-1 transforms
 // like translation, 90-degree rotation, flipping and slicing (by changing
 // strides)
  //  Bitmap(const Bmp::Bitmap<int D, class T> &, int* coordMap, int* translations, int* newlows, int* newwidths);

  ~Bitmap();

  static const int dimensions = Dimensions;  // number of dimensions

 // For every dimension d, the usable coordinate span is (low[d] .. high[d]-1).
  const int low[Dimensions];
  const int high[Dimensions];

  const int width[Dimensions];  // width[d] = high[d] - low[d] (redundant)

 // stride[d] = difference between dimension-d rows (in pixels)
  const int stride[Dimensions];

 // Returns the particular pixel with index i
 // index = x*stride[0] + y*stride[1] + z*stride[2] + ...
  PixelType& operator[] (int index);

// TODO: Returns the pixel with coorinates (x,y,z,...), checks bounds
 //  PixelType& operator() (unsigned row, unsigned col, ...);


};

}

#endif
