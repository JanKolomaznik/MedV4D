#ifndef COORDINATES_H
#define COORDINATES_H

#include "ExceptionBase.h"

template< typename CoordType, unsigned Dim >
class Coordinates;

template< typename CoordType, unsigned Dim >
inline CoordType &
GetCoordinate( Coordinates< CoordType, Dim > &coord, unsigned idx )
{
	if ( idx >= 0 && idx < Dim ) { 
		return coord._coordinates[ idx ];
	} else {
		throw EWrongIndex();
	}
}

template< typename CoordType, unsigned Dim >
inline CoordType
GetConstCoordinate( const Coordinates< CoordType, Dim > &coord, unsigned idx )
{
	if ( idx >= 0 && idx < Dim ) { 
		return coord._coordinates[ idx ];
	} else {
		throw EWrongIndex();
	}
}
template < typename CoordType, unsigned Dimension >
Coordinates< CoordType, Dimension > operator+ ( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
template < typename CoordType, unsigned Dimension >
Coordinates< CoordType, Dimension > operator-( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
template < typename CoordType, unsigned Dimension >
Coordinates< CoordType, Dimension > operator+=( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
template < typename CoordType, unsigned Dimension >
Coordinates< CoordType, Dimension > operator-=( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
template < typename CoordType, unsigned Dimension >
Coordinates< CoordType, Dimension > operator-( const Coordinates< CoordType, Dimension > &v );
template < typename CoordType, unsigned Dimension >
Coordinates< CoordType, Dimension > operator*( CoordType k, const Coordinates< CoordType, Dimension > &v );

template< typename CoordType  >
class Coordinates< CoordType, 1 >
{
public:
	static const unsigned Dimension = 1;

	friend CoordType & GetCoordinate< CoordType, Dimension >( Coordinates< CoordType, Dimension > &coord, unsigned idx );
	friend CoordType GetConstCoordinate< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &coord, unsigned idx );

	friend Coordinates< CoordType, Dimension > operator+ < CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator- < CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator+= < CoordType, Dimension >( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-= < CoordType, Dimension >( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator- < CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v );

	typedef CoordType 	CoordinateType;

	Coordinates()
		{ for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = 0; } }

	Coordinates( const Coordinates< CoordType, Dimension > &coord ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = coord._coordinates[i]; } 
		}

	explicit Coordinates( const CoordType &x )
		{ _coordinates[0] = x; }

	CoordinateType &
	operator[]( unsigned idx )
		{ return GetCoordinate( *this, idx ); }

	CoordinateType &
	operator[]( unsigned idx )const
		{ return GetConstCoordinate( *this, idx ); }

	Coordinates< CoordType, Dimension >
	operator=( const Coordinates< CoordType, Dimension > &coord ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = coord._coordinates[i]; } 
			return *this;
		}
private:
	CoordinateType	_coordinates[ Dimension ];
};

template< typename CoordType  >
class Coordinates< CoordType, 2 >
{
public:
	static const unsigned Dimension = 2;

	friend CoordType & GetCoordinate< CoordType, Dimension >( Coordinates< CoordType, Dimension > &coord, unsigned idx );
	friend CoordType GetConstCoordinate< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &coord, unsigned idx );

	friend Coordinates< CoordType, Dimension > operator+< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator+=< CoordType, Dimension >( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-=< CoordType, Dimension >( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v );

	typedef CoordType 	CoordinateType;

	Coordinates()
		{ for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = 0; } }

	explicit Coordinates( const CoordType &x, const CoordType &y )
		{ 
			_coordinates[0] = x; 
			_coordinates[1] = y;
		}

	Coordinates( const Coordinates< CoordType, Dimension > &coord ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = coord._coordinates[i]; } 
		}

	CoordinateType &
	operator[]( unsigned idx )
		{ return GetCoordinate( *this, idx ); }

	CoordinateType
	operator[]( unsigned idx )const
		{ return GetConstCoordinate( *this, idx ); }

	Coordinates< CoordType, Dimension >
	operator=( const Coordinates< CoordType, Dimension > &coord ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = coord._coordinates[i]; } 
			return *this;
		}
private:
	CoordinateType	_coordinates[ Dimension ];
};

template< typename CoordType  >
class Coordinates< CoordType, 3 >
{
public:
	static const unsigned Dimension = 3;

	friend CoordType & GetCoordinate< CoordType, Dimension >( Coordinates< CoordType, Dimension > &coord, unsigned idx );
	friend CoordType GetConstCoordinate< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &coord, unsigned idx );

	friend Coordinates< CoordType, Dimension > operator+< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator+=< CoordType, Dimension >( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-=< CoordType, Dimension >( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v );

	typedef CoordType 	CoordinateType;

	Coordinates()
		{ for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = 0; } }

	explicit Coordinates( const CoordType &x, const CoordType &y, const CoordType &z )
		{ 
			_coordinates[0] = x; 
			_coordinates[1] = y;
			_coordinates[2] = z;
		}

	Coordinates( const Coordinates< CoordType, Dimension > &coord ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = coord._coordinates[i]; } 
		}

	CoordinateType &
	operator[]( unsigned idx )
		{ return GetCoordinate( *this, idx ); }

	CoordinateType &
	operator[]( unsigned idx )const
		{ return GetConstCoordinate( *this, idx ); }

	Coordinates< CoordType, Dimension >
	operator=( const Coordinates< CoordType, Dimension > &coord ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = coord._coordinates[i]; } 
			return *this;
		}
private:
	CoordinateType	_coordinates[ Dimension ];
};

template< typename CoordType  >
class Coordinates< CoordType, 4 >
{
public:
	static const unsigned Dimension = 4;

	friend CoordType & GetCoordinate< CoordType, Dimension >( Coordinates< CoordType, Dimension > &coord, unsigned idx );
	friend CoordType GetConstCoordinate< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &coord, unsigned idx );

	friend Coordinates< CoordType, Dimension > operator+< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator+=< CoordType, Dimension >( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-=< CoordType, Dimension >( Coordinates< CoordType, Dimension > &v1, const Coordinates< CoordType, Dimension > &v2 );
	friend Coordinates< CoordType, Dimension > operator-< CoordType, Dimension >( const Coordinates< CoordType, Dimension > &v );

	typedef CoordType 	CoordinateType;

	Coordinates()
		{ for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = 0; } }

	Coordinates( const Coordinates< CoordType, Dimension > &coord ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) 
			{ _coordinates[i] = coord._coordinates[i]; } 
		}

	explicit Coordinates( const CoordType &x, const CoordType &y, const CoordType &z, const CoordType &t )
		{ 
			_coordinates[0] = x; 
			_coordinates[1] = y;
			_coordinates[2] = z;
			_coordinates[3] = t;
		}

	CoordinateType &
	operator[]( unsigned idx )
		{ return GetCoordinate( *this, idx ); }

	CoordinateType &
	operator[]( unsigned idx )const
		{ return GetConstCoordinate( *this, idx ); }

	Coordinates< CoordType, Dimension >
	operator=( const Coordinates< CoordType, Dimension > &coord ) 
		{ 
			for( unsigned i=0; i<Dimension; ++i ) { _coordinates[i] = coord._coordinates[i]; } 
			return *this;
		}
private:
	CoordinateType	_coordinates[ Dimension ];
};

template< typename CoordType, unsigned Dim >
Coordinates< CoordType, Dim >
operator+( const Coordinates< CoordType, Dim > &v1, const Coordinates< CoordType, Dim > &v2 )
{
	Coordinates< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result._coordinates[ i ] = v1._coordinates[ i ] + v2._coordinates[ i ];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
Coordinates< CoordType, Dim >
operator-( const Coordinates< CoordType, Dim > &v1, const Coordinates< CoordType, Dim > &v2 )
{
	Coordinates< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result._coordinates[ i ] = v1._coordinates[ i ] - v2._coordinates[ i ];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
Coordinates< CoordType, Dim >
operator+=( Coordinates< CoordType, Dim > &v1, const Coordinates< CoordType, Dim > &v2 )
{
	for( unsigned i=0; i < Dim; ++i ) {
		v1._coordinates[ i ] += v2._coordinates[ i ];
	}
	return v1;
}

template< typename CoordType, unsigned Dim >
Coordinates< CoordType, Dim >
operator-=( Coordinates< CoordType, Dim > &v1, const Coordinates< CoordType, Dim > &v2 )
{
	for( unsigned i=0; i < Dim; ++i ) {
		v1._coordinates[ i ] -= v2._coordinates[ i ];
	}
	return v1;
}

template< /*typename ScalarType, */typename CoordType, unsigned Dim >
Coordinates< CoordType, Dim >
operator*( /*ScalarType*/CoordType k, const Coordinates< CoordType, Dim > &v )
{
	Coordinates< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result[ i ] = k * v[ i ];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
Coordinates< CoordType, Dim >
operator*=( Coordinates< CoordType, Dim > &v, CoordType k )
{
	for( unsigned i=0; i < Dim; ++i ) {
		v[ i ] *= k;
	}

	return v;
}

template< typename CoordType, unsigned Dim >
CoordType
operator*( const Coordinates< CoordType, Dim > &a, const Coordinates< CoordType, Dim > &b )
{
	CoordType result = TypeTraits< CoordType >::Zero; 
	for( unsigned i=0; i < Dim; ++i ) {
		result += a[i] * b[i];
	}

	return result;
}

template< typename CoordType, unsigned Dim >
Coordinates< CoordType, Dim >
operator-( const Coordinates< CoordType, Dim > &v )
{
	Coordinates< CoordType, Dim > result;

	for( unsigned i=0; i < Dim; ++i ) {
		result._coordinates[ i ] = -v[ i ];
	}

	return result;
}

template< typename StreamType, typename CoordType, unsigned Dim >
StreamType &
operator<<( StreamType &stream, const Coordinates< CoordType, Dim > &coords )
{
	for( unsigned i=0; i < Dim; ++i ) {
		stream << coords[i];
		if( i != Dim-1 ) {
			stream << " ";
		}
	}
	return stream;
}

typedef Coordinates< int32, 2 > CoordInt2D;
typedef Coordinates< int32, 3 > CoordInt3D;
typedef Coordinates< int32, 4 > CoordInt4D;

#endif /*COORDINATES_H*/
