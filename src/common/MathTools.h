#ifndef MATH_TOOLS_H
#define MATH_TOOLS_H


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


#endif /*MATH_TOOLS_H*/
