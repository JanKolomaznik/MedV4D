#ifndef MATH_TOOLS_H
#define MATH_TOOLS_H

#include <cmath>
#include "common/Vector.h"

extern const float32 Epsilon;
extern const float32 PI;
extern const float32 PIx2;
extern const float32 PIx3;
extern const float32 PIx4;
extern const float32 PId2;
extern const float32 PId3;
extern const float32 PId4;


//#define Max( a, b ) ((a)<(b) ? (b) : (a))
//#define Min( a, b ) ((a)<(b) ? (a) : (b))
//#define MOD( a, b ) ((a)<0 ? ((a)%(b)) + (b) : (a) % (b))
#define PWR( a ) ( (a) * (a) )
#define ROUND( a ) ( (int)(a+0.5) )

template< typename NTypeA, typename NTypeB >
NTypeB
MOD( NTypeA a, NTypeB b );

template<>
inline int32
MOD( int32 a, int32 b )
{
	int32 val = a % b;
	if( val < 0 ) {
		val += b;
	}
	return val;
}

template<>
inline uint32
MOD( int32 a, uint32 b )
{
	int32 val = a % b;
	if( val < 0 ) {
		val += b;
	}
	return (uint32)val;
}

template<>
inline uint32
MOD( uint32 a, uint32 b )
{
	return a % b;
}

/*template<>
inline int64
MOD( int64 a, int64 b )
{
	if( a < 0 ) {
		return a % b + b;
	}
	return a % b;
}*/

template< typename NType >
inline NType
Max( NType a, NType b ) {
	if( a<b ) return b;

	return a;
}

template< typename NType >
inline NType
Max( NType a, NType b, NType c ) {
	if( a<b ) return Max( b, c );

	return Max( a, c );
}

template< typename NType >
inline NType
Min( NType a, NType b ) {
	if( a>b ) return b;

	return a;
}

template< typename NType >
inline NType
Min( NType a, NType b, NType c ) {
	if( a>b ) return Min( b, c );

	return Min( a, c );
}

template< typename TNumType, size_t tDim >
inline TNumType
Max( const Vector< TNumType, tDim > &a ) {
	TNumType res = a[0];
	for( size_t i = 1; i < tDim; ++i ) {
		if ( res < a[(unsigned int)i] ) {
			res = a[(unsigned int)i];
		}
	}
	return res;
}

template< typename TNumType, size_t tDim >
inline TNumType
Min( const Vector< TNumType, tDim > &a ) {
	TNumType res = a[0];
	for( size_t i = 1; i < tDim; ++i ) {
		if ( res > a[(unsigned int)i] ) {
			res = a[(unsigned int)i];
		}
	}
	return res;
}

template< typename TNumType, size_t tDim >
inline Vector< TNumType, tDim >
Max( const Vector< TNumType, tDim > &a, const Vector< TNumType, tDim > &b ) {
	Vector< TNumType, tDim > res;
	for( size_t i = 0; i < tDim; ++i ) {
		res[i] = Max( a[i], b[i] );
	}
	return res;
}


template< typename TNumType, size_t tDim >
inline Vector< TNumType, tDim >
Min( const Vector< TNumType, tDim > &a, const Vector< TNumType, tDim > &b ) {
	Vector< TNumType, tDim > res;
	for( size_t i = 0; i < tDim; ++i ) {
		res[i] = Min( a[i], b[i] );
	}
	return res;
}

template< typename NType >
inline NType
Abs( NType a ) {
	if( (a)<0 ) return -1 * a;

	return a;
}

template< typename NType >
inline NType
Sqr( NType a ) {
	return a*a;
}

template< typename NType >
inline NType
Sqrt( NType a ) {
	return sqrt( a );
}

template< typename NType >
inline int32
Sgn( NType a ) {
	if( a < 0 ) {
		return -1;
	} 
	if( a > 0 ) {
		return 1;
	} 

	return 0;
}

inline int32
Round( float32 aValue )
{
	return (int)ceil( aValue + 0.5f );
}

inline int32
Round( float64 aValue )
{
	return (int)ceil( aValue + 0.5 );
}

template< size_t tDim >
inline Vector< int32, tDim >
Round( const Vector< float32, tDim > &a ) {
	Vector< int32, tDim > res;
	for( size_t i = 0; i < tDim; ++i ) {
		res[i] = Round( a[i] );
	}
	return res;
}

template< size_t tDim >
inline Vector< int32, tDim >
Round( const Vector< float64, tDim > &a ) {
	Vector< int32, tDim > res;
	for( size_t i = 0; i < tDim; ++i ) {
		res[i] = Round( a[i] );
	}
	return res;
}

template< typename NType >
inline NType
ClampToInterval( NType a, NType b, NType val ) {
	if( val < a ) {
		return a;
	} 
	if( val > b ) {
		return b;
	} 

	return val;
}

inline bool
EpsilonTest( float32 &aValue, float32 aEpsilon = Epsilon )
{
	return Abs( aValue ) < aEpsilon;
}

inline bool
EpsilonTest( float64 &aValue, float64 aEpsilon = Epsilon )
{
	return Abs( aValue ) < aEpsilon;
}



#endif /*MATH_TOOLS_H*/
