#ifndef MATH_TOOLS_H
#define MATH_TOOLS_H

#include <cmath>
#include "MedV4D/Common/Vector.h"

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <cmath>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace M4D
{

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

template< typename TNumType, size_t tDim >
inline TNumType
max( const Vector< TNumType, tDim > &a ) {
	TNumType res = a[0];
	for( size_t i = 1; i < tDim; ++i ) {
		if ( res < a[(unsigned int)i] ) {
			res = a[(unsigned int)i];
		}
	}
	return res;
}

template< typename NType >
inline NType
max( NType a, NType b ) {
	BOOST_STATIC_ASSERT( (boost::is_fundamental< NType >::value) );
	if( a<b ) return b;

	return a;
}

template< typename NType >
inline NType
max( NType a, NType b, NType c ) {
	BOOST_STATIC_ASSERT( (boost::is_fundamental< NType >::value) );
	if( a<b ) return max( b, c );

	return max( a, c );
}

template< typename NType >
inline NType
max( NType a, NType b, NType c, NType d ) {
	BOOST_STATIC_ASSERT( (boost::is_fundamental< NType >::value) );
	return max( max( a, b ), max( c, d ) );
}

template< typename TNumType, size_t tDim >
inline Vector< TNumType, tDim >
maxVect( const Vector< TNumType, tDim > &a, const Vector< TNumType, tDim > &b ) {
	Vector< TNumType, tDim > res;
	for( size_t i = 0; i < tDim; ++i ) {
		res[i] = max( a[i], b[i] );
	}
	return res;
}

template< typename TNumType, size_t tDim >
inline Vector< TNumType, tDim >
maxVect( const Vector< TNumType, tDim > &a, const Vector< TNumType, tDim > &b, const Vector< TNumType, tDim > &c ) {
	return maxVect< TNumType, tDim >( a, maxVect< TNumType, tDim >( b, c ) );
}

template< typename TNumType, size_t tDim >
inline Vector< TNumType, tDim >
maxVect( const Vector< TNumType, tDim > &a, const Vector< TNumType, tDim > &b, const Vector< TNumType, tDim > &c, const Vector< TNumType, tDim > &d ) {
	return maxVect< TNumType, tDim >( maxVect< TNumType, tDim >( a, b ), maxVect< TNumType, tDim >( c, d ) );
}




//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template< typename NType >
inline NType
min( NType a, NType b ) {
	BOOST_STATIC_ASSERT( (boost::is_fundamental< NType >::value) );
	if( a>b ) return b;

	return a;
}

template< typename NType >
inline NType
min( NType a, NType b, NType c ) {
	BOOST_STATIC_ASSERT( (boost::is_fundamental< NType >::value) );
	if( a>b ) return min( b, c );

	return min( a, c );
}

template< typename NType >
inline NType
min( NType a, NType b, NType c, NType d ) {
	BOOST_STATIC_ASSERT( (boost::is_fundamental< NType >::value) );
	return min( min( a, b ), min( c, d ) );
}

template< typename TNumType, size_t tDim >
inline TNumType
min( const Vector< TNumType, tDim > &a ) {
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
minVect( const Vector< TNumType, tDim > &a, const Vector< TNumType, tDim > &b ) {
	Vector< TNumType, tDim > res;
	for( size_t i = 0; i < tDim; ++i ) {
		res[i] = min( a[i], b[i] );
	}
	return res;
}

template< typename TNumType, size_t tDim >
inline Vector< TNumType, tDim >
minVect( const Vector< TNumType, tDim > &a, const Vector< TNumType, tDim > &b, const Vector< TNumType, tDim > &c ) {
	return minVect< TNumType, tDim >( a, minVect< TNumType, tDim >( b, c ) );
}

template< typename TNumType, size_t tDim >
inline Vector< TNumType, tDim >
minVect( const Vector< TNumType, tDim > &a, const Vector< TNumType, tDim > &b, const Vector< TNumType, tDim > &c, const Vector< TNumType, tDim > &d ) {
	return minVect< TNumType, tDim >( minVect< TNumType, tDim >( a, b ), minVect< TNumType, tDim >( c, d ) );
}


//******************************************************************************


template< typename TNumType, size_t tDim >
inline size_t
maxIdx( const Vector< TNumType, tDim > &a ) {
	size_t idx = 0;
	for( size_t i = 1; i < tDim; ++i ) {
		if ( a[idx] < a[(unsigned int)i] ) {
			idx = i;
		}
	}
	return idx;
}

template< typename TNumType, size_t tDim >
inline size_t
minIdx( const Vector< TNumType, tDim > &a ) {
	size_t idx = 0;
	for( size_t i = 1; i < tDim; ++i ) {
		if ( a[idx] > a[i] ) {
			idx = i;
		}
	}
	return idx;
}

template< typename NType >
inline NType
abs( NType a ) {
	if( (a)<0 ) return -1 * a;

	return a;
}

template< typename NType >
inline NType
sqr( NType a ) {
	return a*a;
}

/*template< typename NType >
inline NType
sqrt( NType a ) {
	return sqrt( a );
}*/

template< typename NType >
inline int32
sgn( NType a ) {
	if( a < 0 ) {
		return -1;
	} 
	if( a > 0 ) {
		return 1;
	} 

	return 0;
}



inline int32
floor( float32 aValue )
{
	return (int)floorf( aValue );
}

/*inline int32
round( float64 aValue )
{
	return (int)ceil( aValue + 0.5 );
}*/

inline int32
round( float32 aValue )
{
	return (int)M4D::floor( aValue + 0.5f );
}

template< size_t tDim >
inline Vector< int32, tDim >
round( const Vector< float32, tDim > &a ) {
	Vector< int32, tDim > res;
	for( size_t i = 0; i < tDim; ++i ) {
		res[i] = round( a[i] );
	}
	return res;
}

template< size_t tDim >
inline Vector< int32, tDim >
round( const Vector< float64, tDim > &a ) {
	Vector< int32, tDim > res;
	for( size_t i = 0; i < tDim; ++i ) {
		res[i] = round( a[i] );
	}
	return res;
}

template< size_t tDim >
inline Vector< int32, tDim >
floor( const Vector< float32, tDim > &a ) {
	Vector< int32, tDim > res;
	for( size_t i = 0; i < tDim; ++i ) {
		res[i] = M4D::floor( a[i] );
	}
	return res;
}


template< typename NType >
inline NType
clampToInterval( NType a, NType b, NType val ) {
	if( val < a ) {
		return a;
	} 
	if( val > b ) {
		return b;
	} 

	return val;
}

template< typename NType >
inline bool
intervalTest( NType a, NType b, NType val ) {
	if( val < a ) {
		return false;
	} 
	if( val > b ) {
		return false;
	} 

	return true;
}

inline bool
epsilonTest( float32 &aValue, float32 aEpsilon = Epsilon )
{
	return abs( aValue ) < aEpsilon;
}

inline bool
epsilonTest( float64 &aValue, float64 aEpsilon = Epsilon )
{
	return abs( aValue ) < aEpsilon;
}

}//namespace M4D


#endif /*MATH_TOOLS_H*/
