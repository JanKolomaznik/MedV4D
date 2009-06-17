#ifndef COORDINATES_H
#define COORDINATES_H

#include "common/ExceptionBase.h"
#include <istream>

#ifdef DEBUG_LEVEL
	#define CHECK_INDICES_ENABLED
#endif /*DEBUG_LEVEL*/

template< typename CoordType, unsigned Dim >
class Vector;

template < typename CoordType, unsigned Dimension >
Vector< CoordType, Dimension > operator+ ( const Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
template < typename CoordType, unsigned Dimension >
Vector< CoordType, Dimension > operator-( const Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
template < typename CoordType, unsigned Dimension >
Vector< CoordType, Dimension > operator+=( Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
template < typename CoordType, unsigned Dimension >
Vector< CoordType, Dimension > operator-=( Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
template < typename CoordType, unsigned Dimension >
Vector< CoordType, Dimension > operator-( const Vector< CoordType, Dimension > &v );
template < typename CoordType, unsigned Dimension >
Vector< CoordType, Dimension > operator*( CoordType k, const Vector< CoordType, Dimension > &v );


template< typename CoordType, unsigned Dim >
class Vector
{
public:
	static const unsigned Dimension = Dim;

	friend Vector< CoordType, Dimension > operator+< CoordType, Dimension >( const Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
	friend Vector< CoordType, Dimension > operator-< CoordType, Dimension >( const Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
	friend Vector< CoordType, Dimension > operator+=< CoordType, Dimension >( Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
	friend Vector< CoordType, Dimension > operator-=< CoordType, Dimension >( Vector< CoordType, Dimension > &v1, const Vector< CoordType, Dimension > &v2 );
	friend Vector< CoordType, Dimension > operator-< CoordType, Dimension >( const Vector< CoordType, Dimension > &v );

	typedef CoordType 	CoordinateType;

	Vector()
		{ for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = 0; } }

	explicit Vector( CoordType x )
		{
			for( unsigned i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = x; } 
		}

	Vector( CoordType x, CoordType y )
		{
			if( Dimension != 2 ) { 
				_THROW_ M4D::ErrorHandling::EBadDimension();
			}

			_coordinates[0] = x; 
			_coordinates[1] = y; 
		}

	Vector( CoordType x, CoordType y, CoordType z )
		{
			if( Dimension != 3 ) {
				_THROW_ M4D::ErrorHandling::EBadDimension();
			}

			_coordinates[0] = x; 
			_coordinates[1] = y; 
			_coordinates[2] = z; 
		}

	Vector( CoordType x, CoordType y, CoordType z , CoordType w )
		{
			if( Dimension != 4 ) {
				_THROW_ M4D::ErrorHandling::EBadDimension();
			}

			_coordinates[0] = x; 
			_coordinates[1] = y; 
			_coordinates[2] = z; 
			_coordinates[3] = w; 
		}
	

	template< typename CType >
	Vector( const Vector< CType, Dimension > &coord ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = coord[i]; } 
		}

	Vector( const CoordType coords[] ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = coords[i]; } 
		}

	CoordinateType &
	operator[]( unsigned idx )
		{ 
#ifdef CHECK_INDICES_ENABLED
			if ( idx >= Dim ) { 
				_THROW_ M4D::ErrorHandling::EBadIndex();
			}
#endif /*CHECK_INDICES_ENABLED*/

			return _coordinates[ idx ];
		}

	CoordinateType
	operator[]( unsigned idx )const
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
			for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = coord[i]; } 
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
		for( unsigned i=0; i<Dimension; ++i ) { 
			tmp = _coordinates[i];
			//BINSTREAM_WRITE_MACRO( stream, tmp );
			stream.write( (char*)&tmp, sizeof(tmp) );
		}
	}
	void
	FromBinStream( std::istream &stream )
	{
		CoordinateType tmp;
		for( unsigned i=0; i<Dimension; ++i ) { 
			//BINSTREAM_READ_MACRO( stream, tmp );
			stream.read( (char*)&tmp, sizeof(tmp) );
			_coordinates[i] = tmp;
		}
	}
private:
	CoordinateType	_coordinates[ Dimension ];
};


template< typename CoordType, unsigned Dim >
bool
operator==( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	bool result = true;

	for( unsigned i=0; i < Dim; ++i ) {
		result = result && (v1[ i ] == v2[ i ]);
	}

	return result;
}


template< typename CoordType, unsigned Dim >
bool
operator!=( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	bool result = false;

	for( unsigned i=0; i < Dim; ++i ) {
		result = result || (v1[ i ] != v2[ i ]);
	}

	return result;
}


template< typename CoordType, unsigned Dim >
bool
operator<=( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	bool result = true;
	for( unsigned i=0; i < Dim; ++i ) {
		result = result && (v1[ i ] <= v2[ i ]);
	}

	return result;
}

template< typename CoordType, unsigned Dim >
bool
operator<( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	bool result = true;
	for( unsigned i=0; i < Dim; ++i ) {
		result = result && (v1[ i ] < v2[ i ]);
	}

	return result;
}



template< typename CoordType, unsigned Dim >
bool
operator>=( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	return v2 <= v1;
}

template< typename CoordType, unsigned Dim >
bool
operator>( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	return v2 < v1;
}

template< typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
operator+( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	Vector< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result._coordinates[ i ] = v1._coordinates[ i ] + v2._coordinates[ i ];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
operator-( const Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	Vector< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result._coordinates[ i ] = v1._coordinates[ i ] - v2._coordinates[ i ];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
operator+=( Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	for( unsigned i=0; i < Dim; ++i ) {
		v1._coordinates[ i ] += v2._coordinates[ i ];
	}
	return v1;
}

template< typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
operator-=( Vector< CoordType, Dim > &v1, const Vector< CoordType, Dim > &v2 )
{
	for( unsigned i=0; i < Dim; ++i ) {
		v1._coordinates[ i ] -= v2._coordinates[ i ];
	}
	return v1;
}

template< /*typename ScalarType, */typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
operator*( /*ScalarType*/CoordType k, const Vector< CoordType, Dim > &v )
{
	Vector< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result[ i ] = k * v[ i ];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
operator*( const Vector< CoordType, Dim > &v, CoordType k )
{
	Vector< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result[ i ] = k * v[ i ];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
operator*=( Vector< CoordType, Dim > &v, CoordType k )
{
	for( unsigned i=0; i < Dim; ++i ) {
		v[ i ] *= k;
	}

	return v;
}

template< typename CoordType, unsigned Dim >
CoordType
operator*( const Vector< CoordType, Dim > &a, const Vector< CoordType, Dim > &b )
{
	CoordType result = TypeTraits< CoordType >::Zero; 
	for( unsigned i=0; i < Dim; ++i ) {
		result += a[i] * b[i];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
operator-( const Vector< CoordType, Dim > &v )
{
	Vector< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result._coordinates[ i ] = -v[ i ];
	}

	return result;
}

template< typename StreamType, typename CoordType, unsigned Dim >
StreamType &
operator<<( StreamType &stream, const Vector< CoordType, Dim > &coords )
{
	for( unsigned i=0; i < Dim; ++i ) {
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
template< typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
VectorDimensionsShiftRight( const Vector< CoordType, Dim > &v )
{
	Vector< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result[ (i + 1) % Dim ] = v[ i ];
	}

	return result;
}

/**
 * Will shift coordinate in every dimension to dimesion on left, 
 * first coordinate will become last.
 **/
template< typename CoordType, unsigned Dim >
Vector< CoordType, Dim >
VectorDimensionsShiftLeft( const Vector< CoordType, Dim > &v )
{
	Vector< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result[ i ] = v[ (i + 1) % Dim ];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
void
VectorAbs( Vector< CoordType, Dim > &v )
{
	for( unsigned i=0; i < Dim; ++i ) {
		v[ i ] = Abs( v[ i ] );
	}
}

typedef Vector< int32, 2 > CoordInt2D;
typedef Vector< int32, 3 > CoordInt3D;
typedef Vector< int32, 4 > CoordInt4D;

typedef Vector< int32, 2 >			RasterPos;

typedef Vector< uint32, 2 > Size2D;
typedef Vector< uint32, 3 > Size3D;

#endif /*COORDINATES_H*/
