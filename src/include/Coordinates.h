#ifndef COORDINATES_H
#define COORDINATES_H

#include "ExceptionBase.h"

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

template< typename CoordType, unsigned Dim >
class Coordinates;

template< typename CoordType  >
class Coordinates< CoordType, 1 >
{
public:
	friend CoordType & GetCoordinate( Coordinates< CoordType, Dim > &coord, unsigned idx );
	friend CoordType & GetConstCoordinate( const Coordinates< CoordType, Dim > &coord, unsigned idx );
	static const unsigned Dimension = 1;
	typedef CoordType 	CoordinateType;

	Coordinates( const CoordType &x )
		{ _coordinates[0] = x; }

	CoordinateType &
	operator[]( unsigned idx )
		{ return GetCoordinate( *this, idx ); }

	CoordinateType &
	operator[]( unsigned idx )const
		{ return GetConstCoordinate( *this, idx ); }
private:
	CoordinateType	_coordinates[ Dimension ];
};

template< typename CoordType  >
class Coordinates< CoordType, 2 >
{
public:
	friend CoordType & GetCoordinate( Coordinates< CoordType, Dim > &coord, unsigned idx );
	friend CoordType & GetConstCoordinate( const Coordinates< CoordType, Dim > &coord, unsigned idx );
	static const unsigned Dimension = 2;
	typedef CoordType 	CoordinateType;

	Coordinates( const CoordType &x, const CoordType &y );
		{ 
			_coordinates[0] = x; 
			_coordinates[1] = y;
		}

	CoordinateType &
	operator[]( unsigned idx )
		{ return GetCoordinate( *this, idx ); }

	CoordinateType &
	operator[]( unsigned idx )const
		{ return GetConstCoordinate( *this, idx ); }
};

template< typename CoordType  >
class Coordinates< CoordType, 3 >
{
public:
	friend CoordType & GetCoordinate( Coordinates< CoordType, Dim > &coord, unsigned idx );
	friend CoordType & GetConstCoordinate( const Coordinates< CoordType, Dim > &coord, unsigned idx );
	static const unsigned Dimension = 3;
	typedef CoordType 	CoordinateType;

	Coordinates( const CoordType &x, const CoordType &y, const CoordType &z );
		{ 
			_coordinates[0] = x; 
			_coordinates[1] = y;
			_coordinates[2] = z;
		}

	CoordinateType &
	operator[]( unsigned idx )
		{ return GetCoordinate( *this, idx ); }

	CoordinateType &
	operator[]( unsigned idx )const
		{ return GetConstCoordinate( *this, idx ); }
};

template< typename CoordType  >
class Coordinates< CoordType, 4 >
{
public:
	friend CoordType & GetCoordinate( Coordinates< CoordType, Dim > &coord, unsigned idx );
	friend CoordType & GetConstCoordinate( const Coordinates< CoordType, Dim > &coord, unsigned idx );
	static const unsigned Dimension = 4;
	typedef CoordType 	CoordinateType;

	Coordinates( const CoordType &x, const CoordType &y, const CoordType &z, const CoordType &t );
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
};

#endif /*COORDINATES_H*/
