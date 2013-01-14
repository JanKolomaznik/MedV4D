#ifndef COORDINATES_H
#define COORDINATES_H

#include "MedV4D/Common/Debug.h"
#include "MedV4D/Common/ExceptionBase.h"
#include <istream>
#include "MedV4D/Common/TypeTraits.h"
#include "MedV4D/Common/TypeComparator.h"

#include <boost/static_assert.hpp>

#ifdef DEBUG_LEVEL
	#define CHECK_INDICES_ENABLED
#endif /*DEBUG_LEVEL*/

template< typename CoordType, size_t Dim >
class Vector;

template < typename CoordType, size_t Dimension >
Vector< CoordType, Dimension > operator+ ( const Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
template < typename CoordType, size_t Dimension >
Vector< CoordType, Dimension > operator-( const Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
template < typename CoordType, size_t Dimension >
Vector< CoordType, Dimension > operator+=( Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
template < typename CoordType, size_t Dimension >
Vector< CoordType, Dimension > operator-=( Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
template < typename CoordType, size_t Dimension >
Vector< CoordType, Dimension > operator-( const Vector< CoordType, Dimension > &v );
template < typename CoordType, size_t Dimension >
Vector< CoordType, Dimension > operator*( CoordType k, const Vector< CoordType, Dimension > &v );

typedef size_t VectorDim_t;

template< typename CoordType, size_t Dim >
class Vector
{
public:
	static const size_t Dimension = Dim;

	friend Vector< CoordType, Dimension > operator+< CoordType, Dimension >( const Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
	friend Vector< CoordType, Dimension > operator-< CoordType, Dimension >( const Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
	friend Vector< CoordType, Dimension > operator+=< CoordType, Dimension >( Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
	friend Vector< CoordType, Dimension > operator-=< CoordType, Dimension >( Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
	friend Vector< CoordType, Dimension > operator-< CoordType, Dimension >( const Vector< CoordType, Dimension > &v );

	typedef CoordType 	CoordinateType;

	template< size_t tCoord >
	struct ValueAccessor
	{
		CoordType &
		operator()( Vector< CoordType, Dimension > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
		CoordType
		operator()( const Vector< CoordType, Dimension > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
	};

	Vector()
		{ for( size_t i=0; i<Dimension; ++i ) { _coordinates[i] = 0; } }

	explicit Vector( CoordType x )
		{
			for( size_t i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = x; } 
		}
	template< typename CType >
	Vector( CType x, CType y )
		{
			/*if( Dimension != 2 ) { 
				_THROW_ M4D::ErrorHandling::EBadDimension();
			}*/
			BOOST_STATIC_ASSERT( Dim == 2 );

			_coordinates[0] = static_cast< CoordType >( x ); 
			_coordinates[1] = static_cast< CoordType >( y ); 
		}

	Vector( CoordType x, CoordType y, CoordType z )
		{
			/*if( Dimension != 3 ) {
				_THROW_ M4D::ErrorHandling::EBadDimension();
			}*/
			BOOST_STATIC_ASSERT( Dim == 3 );

			_coordinates[0] = x; 
			_coordinates[1] = y; 
			_coordinates[2] = z; 
		}

	Vector( CoordType x, CoordType y, CoordType z , CoordType w )
		{
			/*if( Dimension != 4 ) {
				_THROW_ M4D::ErrorHandling::EBadDimension();
			}*/
			BOOST_STATIC_ASSERT( Dim == 4 );

			_coordinates[0] = x; 
			_coordinates[1] = y; 
			_coordinates[2] = z; 
			_coordinates[3] = w; 
		}

	Vector( CoordType p0, CoordType p1, CoordType p2, CoordType p3, CoordType p4 )
		{
			BOOST_STATIC_ASSERT( Dim == 5 );

			_coordinates[0] = p0; 
			_coordinates[1] = p1; 
			_coordinates[2] = p2; 
			_coordinates[3] = p3; 
			_coordinates[4] = p4; 
		}

	Vector( CoordType p0, CoordType p1, CoordType p2, CoordType p3, CoordType p4, CoordType p5 )
		{
			BOOST_STATIC_ASSERT( Dim == 6 );

			_coordinates[0] = p0; 
			_coordinates[1] = p1; 
			_coordinates[2] = p2; 
			_coordinates[3] = p3; 
			_coordinates[4] = p4; 
			_coordinates[5] = p5; 
		}
	

	template< typename CType >
	Vector( const Vector< CType, Dimension > &coord ) 
		{ 
			for( size_t i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = static_cast<CoordType>(coord[i]); } 
		}

	template< typename CType >
	Vector( const Vector< CType, Dimension-1 > &coord, CType value ) 
		{ 
			for( size_t i=0; i<Dimension-1; ++i ) 
			{ _coordinates[i] = static_cast<CoordType>(coord[i]); } 
			_coordinates[Dimension-1] = value;
		}

	Vector( const CoordType coords[] ) 
		{ 
			for( size_t i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = coords[i]; } 
		}

	CoordinateType &
	operator[]( size_t idx )
		{ 
			return Get( idx );
		}

	CoordinateType
	operator[]( size_t idx )const
		{ 
			return Get( idx );
		}


	template< size_t tIdx >
	CoordinateType &
	StaticGet()
		{ 
			BOOST_STATIC_ASSERT( tIdx < Dimension );
			return _coordinates[ tIdx ];
		}


	template< size_t tIdx >
	CoordinateType
	StaticGet()const
		{ 
			BOOST_STATIC_ASSERT( tIdx < Dimension );
			return _coordinates[ tIdx ];
		}

	template< size_t tBegin, size_t tEnd >
	Vector< CoordinateType, tEnd - tBegin >
	GetSubVector() const
	{
		BOOST_STATIC_ASSERT( tBegin < tEnd );
		BOOST_STATIC_ASSERT( Dimension >= tEnd );
		Vector< CoordinateType, tEnd - tBegin > tmp;
		for( size_t i = tBegin; i < tEnd; ++i ) {
			tmp[i-tBegin] = _coordinates[i];
		}
		return tmp;
	}

	CoordinateType &
	Get( size_t idx )
		{ 
#ifdef CHECK_INDICES_ENABLED
			if ( idx >= Dim ) { 
				_THROW_ M4D::ErrorHandling::EBadIndex();
			}
#endif /*CHECK_INDICES_ENABLED*/

			return _coordinates[ idx ];
		}

	CoordinateType
	Get( size_t idx )const
		{ 
#ifdef CHECK_INDICES_ENABLED
			if ( idx >= Dim ) { 
				_THROW_ M4D::ErrorHandling::EBadIndex();
			}
#endif /*CHECK_INDICES_ENABLED*/

			return _coordinates[ idx ];
		}

	template< typename CType >
	Vector< CoordType, Dimension > &
	operator=( const Vector< CType, Dimension > &coord )
		{ 
			for( size_t i=0; i<Dimension; ++i ) { _coordinates[i] = static_cast<CoordType>(coord[i]); } 
			return *this;
		}

	Vector< CoordType, Dimension > &
	operator=( const CoordType coord[] )
		{ 
			for( size_t i=0; i<Dimension; ++i ) { _coordinates[i] = coord[i]; } 
			return *this;
		}

	const CoordinateType*
	GetData()const
		{
			return _coordinates;
		}

	void
	ToBinStream( std::ostream &stream )
	{
		CoordinateType tmp;
		for( size_t i=0; i<Dimension; ++i ) { 
			tmp = _coordinates[i];
			//BINSTREAM_WRITE_MACRO( stream, tmp );
			stream.write( (char*)&tmp, sizeof(tmp) );
		}
	}
	void
	FromBinStream( std::istream &stream )
	{
		CoordinateType tmp;
		for( size_t i=0; i<Dimension; ++i ) { 
			//BINSTREAM_READ_MACRO( stream, tmp );
			stream.read( (char*)&tmp, sizeof(tmp) );
			_coordinates[i] = tmp;
		}
	}
private:
	CoordinateType	_coordinates[ Dimension ];
};

template< typename NumType, size_t Dim >
struct TypeTraits< Vector<NumType, Dim> >
{
	typedef	Vector<NumType, Dim>	Type;
	static const int16	NTID = TypeTraits< NumType >::NTID + NTID_VECTOR_DIM_STEP * Dim;

	static const bool	Signed = TypeTraits< NumType >::Signed;
	static const uint16	BitCount = sizeof( Type )*8;
	static Type		Max;
	static Type		Min;
	static Type		Zero;
	static NumType		One;
	static Type		CentralValue;

	typedef Vector< typename TypeTraits< NumType >::SignedClosestType, Dim > 	SignedClosestType;
	typedef Vector< typename TypeTraits< NumType >::SuperiorType, Dim > 		SuperiorType;
	typedef Vector< typename TypeTraits< NumType >::SuperiorSignedType, Dim >	SuperiorSignedType;
	typedef Vector< typename TypeTraits< NumType >::SuperiorFloatType, Dim >	SuperiorFloatType;

	static std::string
	Typename()
	{
		return TO_STRING( "Vector : " << Dim << " x " << TypeTraits< NumType >::Typename() );
	}
};


template< typename CoordType, size_t Dim >
bool
operator==( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	bool result = true;

	for( size_t i=0; i < Dim; ++i ) {
		result = result && (v1[ i ] == v2[ i ]);
	}

	return result;
}


template< typename CoordType, size_t Dim >
bool
operator!=( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	bool result = false;

	for( size_t i=0; i < Dim; ++i ) {
		result = result || (v1[ i ] != v2[ i ]);
	}

	return result;
}


template< typename CoordType, size_t Dim >
bool
operator<=( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	bool result = true;
	for( size_t i=0; i < Dim; ++i ) {
		result = result && (v1[ i ] <= v2[ i ]);
	}

	return result;
}

template< typename CoordType, size_t Dim >
bool
operator<( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	bool result = true;
	for( size_t i=0; i < Dim; ++i ) {
		result = result && (v1[ i ] < v2[ i ]);
	}

	return result;
}



template< typename CoordType, size_t Dim >
bool
operator>=( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	return v2 <= v1;
}

template< typename CoordType, size_t Dim >
bool
operator>( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	return v2 < v1;
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator+( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	Vector< CoordType, Dim > result;

	for( size_t i=0; i < Dim; ++i ) {
		result._coordinates[ i ] = v1._coordinates[ i ] + v2._coordinates[ i ];
	}

	return result;
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator-( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	Vector< CoordType, Dim > result;

	for( size_t i=0; i < Dim; ++i ) {
		result._coordinates[ i ] = v1._coordinates[ i ] - v2._coordinates[ i ];
	}

	return result;
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator+=( Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	for( size_t i=0; i < Dim; ++i ) {
		v1._coordinates[ i ] += v2._coordinates[ i ];
	}
	return v1;
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator-=( Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	for( size_t i=0; i < Dim; ++i ) {
		v1._coordinates[ i ] -= v2._coordinates[ i ];
	}
	return v1;
}

template< /*typename ScalarType, */typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator*( /*ScalarType*/CoordType k, const Vector< CoordType, Dim > &v )
{
	Vector< CoordType, Dim > result;

	for( size_t i=0; i < Dim; ++i ) {
		result[ i ] = k * v[ i ];
	}

	return result;
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator*( const Vector< CoordType, Dim > &v, CoordType k )
{
	Vector< CoordType, Dim > result;

	for( size_t i=0; i < Dim; ++i ) {
		result[ i ] = k * v[ i ];
	}

	return result;
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator*=( Vector< CoordType, Dim > &v, CoordType k )
{
	for( size_t i=0; i < Dim; ++i ) {
		v[ i ] *= k;
	}

	return v;
}

template< typename CoordType, size_t Dim >
CoordType
operator*( const Vector< CoordType, Dim > &a, const Vector< CoordType, Dim > &b )
{
	CoordType result = TypeTraits< CoordType >::Zero; 
	for( size_t i=0; i < Dim; ++i ) {
		result += a[i] * b[i];
	}

	return result;
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator-( const Vector< CoordType, Dim > &v )
{
	Vector< CoordType, Dim > result;

	for( size_t i=0; i < Dim; ++i ) {
		result._coordinates[ i ] = -v[ i ];
	}

	return result;
}

template<typename CoordType, size_t Dim >
std::ostream &
operator<<( std::ostream &stream, const Vector< CoordType, Dim > &coords )
{
	for( size_t i=0; i < Dim; ++i ) {
		stream << coords[i];
		if( i != Dim-1 ) {
			stream << " ";
		}
	}
	return stream;
}

template< typename CoordType >
Vector< CoordType, 2 >
CreateVector( CoordType x, CoordType y )
{
	CoordType tmp[2];
	tmp[0] = x;
	tmp[1] = y;
	return Vector< CoordType, 2 >( tmp );
}

template< typename CoordType >
Vector< CoordType, 3 >
CreateVector( CoordType x, CoordType y, CoordType z )
{
	CoordType tmp[3];
	tmp[0] = x;
	tmp[1] = y;
	tmp[2] = z;
	return Vector< CoordType, 3 >( tmp );
}


template< typename CoordType >
Vector< CoordType, 4 >
CreateVector( CoordType x, CoordType y, CoordType z, CoordType w )
{
	CoordType tmp[4];
	tmp[0] = x;
	tmp[1] = y;
	tmp[2] = z;
	tmp[3] = w;
	return Vector< CoordType, 4 >( tmp );
}

/**
 * Will shift coordinate in every dimension to dimesion on right, 
 * last coordinate will become first.
 **/
template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
VectorDimensionsShiftRight( const Vector< CoordType, Dim > &v )
{
	Vector< CoordType, Dim > result;

	for( size_t i=0; i < Dim; ++i ) {
		result[ (i + 1) % Dim ] = v[ i ];
	}

	return result;
}

/**
 * Will shift coordinate in every dimension to dimesion on left, 
 * first coordinate will become last.
 **/
template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
VectorDimensionsShiftLeft( const Vector< CoordType, Dim > &v )
{
	Vector< CoordType, Dim > result;

	for( size_t i=0; i < Dim; ++i ) {
		result[ i ] = v[ (i + 1) % Dim ];
	}

	return result;
}

template< typename CoordType, size_t Dim >
void
VectorAbs( Vector< CoordType, Dim > &v )
{
	for( size_t i=0; i < Dim; ++i ) {
		v[ i ] = abs( v[ i ] );
	}
}

template< typename CoordType, size_t Dim >
void
VectorNormalization( Vector< CoordType, Dim > &v )
{
	CoordType size = VectorSize( v );
	size = 1.0f /size;
	v *= size;
}

template< typename CoordType >
Vector< CoordType, 3 >
VectorProduct( const Vector< CoordType, 3 > &a, const Vector< CoordType, 3 > &b )
{
	return Vector< CoordType, 3 >( a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] );
}

template< typename CoordType, size_t Dim >
CoordType
VectorSize( const Vector< CoordType, Dim > &v )
{
	CoordType size = v[0] * v[0];
	for( size_t i=1; i < Dim; ++i ) {
		size += v[ i ] * v[ i ];
	}
	size = sqrt( size );
	return size;
}

template< typename CoordType, size_t Dim >
CoordType
VectorDistance( const Vector< CoordType, Dim > &a, const Vector< CoordType, Dim > &b )
{
	return VectorSize( a - b );
}

template< typename CoordType, size_t Dim >
void
Ortogonalize( const Vector< CoordType, Dim > &a, Vector< CoordType, Dim > &b )
{
	CoordType product = (a * b) /  (a * a);
	
	b -= product * a ;
}

/**
 * \param u Normalized vector, into which the second will be projected.
 **/
template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
VectorProjection( const Vector< CoordType, Dim > &u, const Vector< CoordType, Dim > &v )
{
	return Vector< CoordType, Dim >( (v*u) * u );
}

template< typename CoordType, size_t Dim >
CoordType
VectorCoordinateProduct( const Vector< CoordType, Dim > &v )
{
	CoordType result = v[0];
	for( size_t i=1; i < Dim; ++i ) {
		result *= v[ i ];
	}
	return result;
}

template< typename CoordType1, typename CoordType2, size_t Dim >
Vector< typename TypeComparator< CoordType1, CoordType2 >::Superior, Dim >
VectorMemberProduct( const Vector< CoordType1, Dim > &a, const Vector< CoordType2, Dim > &b )
{
	Vector< typename TypeComparator< CoordType1, CoordType2 >::Superior, Dim > result = a;
	for( size_t i=0; i < Dim; ++i ) {
		result[i] *= b[ i ];
	}
	return result;
}

template< typename CoordType1, typename CoordType2, size_t Dim >
Vector< typename TypeComparator< CoordType1, CoordType2 >::Superior, Dim >
VectorMemberDivision( const Vector< CoordType1, Dim > &a, const Vector< CoordType2, Dim > &b )
{
	Vector< typename TypeComparator< CoordType1, CoordType2 >::Superior, Dim > result = a;
	for( size_t i=0; i < Dim; ++i ) {
		result[i] /= b[ i ];
	}
	return result;
}


template< typename CoordType, size_t Dim >
Vector< CoordType, Dim-1 >
VectorPurgeDimension( const Vector< CoordType, Dim > &u, size_t purgedDimension = Dim-1 )
{
	ASSERT_INFO( purgedDimension < Dim, "Must be valid dimension." );

	CoordType data[Dim-1];
	size_t j = 0;
	for( size_t i=0; i < Dim; ++i ) {
		if( i != purgedDimension ) {
			data[j++] = u[i];
		}
	}
	return Vector< CoordType, Dim-1 >( data );
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim+1 >
VectorInsertDimension( const Vector< CoordType, Dim > &u, CoordType value, size_t insertedDimension = Dim )
{
	ASSERT_INFO( insertedDimension <= Dim, "Must be valid dimension." );

	CoordType data[Dim+1];
	size_t j = 0;
	for( size_t i=0; i <= Dim; ++i ) {
		if( i != insertedDimension ) {
			data[i] = u[j++];
		} else {
			data[i] = value;
		}
	}
	return Vector< CoordType, Dim+1 >( data );
}
/**
 * \param min Represent closed N-dimensional interval together with max.
 * \param max Represent closed N-dimensional interval together with min.
 **/
template< typename CoordType, size_t Dim >
bool
VectorProjectionToInterval( Vector< CoordType, Dim > &v, const Vector< CoordType, Dim > &min, const Vector< CoordType, Dim > &max )
{
	bool result = false;
	for( size_t i=0; i < Dim; ++i ) {
		if ( v[i] < min[i] ) {
			result |= true;
			v[i] = min[i];
		}
		if ( v[i] > max[i] ) {
			result |= true;
			v[i] = max[i];
		}
	} 
	return result;
}

typedef Vector< int32, 2 > CoordInt2D;
typedef Vector< int32, 3 > CoordInt3D;
typedef Vector< int32, 4 > CoordInt4D;

typedef Vector< int32, 2 >			RasterPos;

typedef Vector< uint32, 2 > Size2D;
typedef Vector< uint32, 3 > Size3D;

typedef Vector< unsigned int, 2>	Vector2u;
typedef Vector< unsigned int, 3>	Vector3u;
typedef Vector< unsigned int, 4>	Vector4u;

typedef Vector< int, 2>			Vector2i;
typedef Vector< int, 3>			Vector3i;
typedef Vector< int, 4>			Vector4i;

typedef Vector< float, 2>		Vector2f;
typedef Vector< float, 3>		Vector3f;
typedef Vector< float, 4>		Vector4f;

typedef Vector< double, 2>		Vector2d;
typedef Vector< double, 3>		Vector3d;
typedef Vector< double, 4>		Vector4d;

#endif /*COORDINATES_H*/
