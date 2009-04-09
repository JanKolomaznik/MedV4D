#ifndef COMMONTYPES_H_
#define COMMONTYPES_H_

namespace M4D {
namespace Cell {

typedef float32 TPixelValue;
#define DIM 3

typedef double  TimeStepType;
/** Type used for storing status information */
typedef signed char StatusType;

template<typename T>
struct MyVector
{
	typedef T TValue;
	T data[DIM];
	T & operator[]( unsigned idx )	{ return data[ idx ]; }
	T operator[]( unsigned idx ) const	{ return data[ idx ]; }
};

typedef MyVector<int32> TOffset;
typedef MyVector<uint32> TRadius;
typedef MyVector<uint32>  TIndex;
typedef MyVector<float32> TNeighborhoodScales;

typedef MyVector<uint32> TSize;
typedef MyVector<uint32> TStrides;
typedef MyVector<float32> TContinuousIndex;
typedef MyVector<float32> TSpacing;

void ComputeStridesFromSize(const TSize &size, TStrides &strides);

struct TRegion
{
	TIndex offset;
	TSize size;
};
struct TImageProperties
{
	TRegion region;
	TSpacing spacing;
	TPixelValue *imageData;
};

class SparseFieldLevelSetNode
{
public:
	TIndex               m_Value;
	SparseFieldLevelSetNode *Next;
	SparseFieldLevelSetNode *Previous;
};

}
}  // namespace

#endif /*COMMONTYPES_H_*/
