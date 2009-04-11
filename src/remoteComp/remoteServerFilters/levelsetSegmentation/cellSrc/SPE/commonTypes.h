#ifndef COMMONTYPES_H_
#define COMMONTYPES_H_

namespace M4D {
namespace Cell {

typedef float32 TPixelValue;
#define DIM 3
#define NUM_LAYERS 3
// NUM_LAYERS inward and NUM_LAYERS outward + active layer
#define LYERCOUNT ((NUM_LAYERS * 2) + 1)

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
typedef MyVector<int32>  TIndex;
typedef MyVector<float32> TNeighborhoodScales;

typedef MyVector<uint32> TSize;
typedef MyVector<uint32> TStrides;
typedef MyVector<float32> TContinuousIndex;
typedef MyVector<float32> TSpacing;

void ComputeStridesFromSize(const TSize &size, TStrides &strides);
TIndex operator+(const TIndex &i, const TOffset &o);

template<typename T>
std::ostream &operator<<(std::ostream &s, const MyVector<T> &v)
{
	s << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
	return s;
}

struct TRegion
{
	TIndex offset;
	TSize size;
};

template<typename PixelType>
struct TImageProperties
{
	TRegion region;
	TSpacing spacing;
	PixelType *imageData;
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
