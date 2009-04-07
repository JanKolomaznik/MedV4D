#ifndef COMMONTYPES_H_
#define COMMONTYPES_H_

namespace M4D {
namespace Cell {

typedef float32 TPixelValue;
#define DIM 3

typedef double  TimeStepType;
/** Type used for storing status information */
typedef signed char StatusType;

struct UIntVector
{
	typedef uint32 TValue;
	uint32 data[DIM];
	uint32 & operator[]( unsigned idx )	{ return data[ idx ]; }
	uint32 operator[]( unsigned idx ) const	{ return data[ idx ]; }
};
struct FloatVector
{
	typedef float32 TValue;
	float32 data[DIM];
	float32 & operator[]( unsigned idx )	{ return data[ idx ]; }
	float32 operator[]( unsigned idx ) const	{ return data[ idx ]; }
};
struct IntVector
{
	typedef int32 TValue;
	int32 data[DIM];
	int32 & operator[]( unsigned idx )	{ return data[ idx ]; }
	int32 operator[]( unsigned idx ) const	{ return data[ idx ]; }
};

typedef IntVector TOffset;
typedef UIntVector TRadius;
typedef UIntVector  TIndex;
typedef FloatVector TNeighborhoodScales;

typedef UIntVector TRadius;
typedef UIntVector TSize;
typedef UIntVector TStrides;
typedef FloatVector TContinuousIndex;
typedef FloatVector TSpacing;

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
