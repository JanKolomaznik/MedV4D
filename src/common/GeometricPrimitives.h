#ifndef GEOMETRIC_PRIMITIVES_H
#define GEOMETRIC_PRIMITIVES_H

#include "common/Vector.h"

namespace M4D
{

template< typename TType >
class Point2D: public Vector< TType, 2 >
{
public:
	typedef Vector< TType, 2 > PredecessorType;

	Point2D(){}

	explicit Point2D( const PredecessorType &aVec ): PredecessorType( aVec )
	{}

	Point2D( const TType &aX, const TType &aY ): PredecessorType( aX, aY )
	{}

	TType &
	x() { return PredecessorType::template StaticGet< 0 >(); }

	const TType &
	x() const { return PredecessorType::template StaticGet< 0 >(); }

	TType &
	y() { return PredecessorType::template StaticGet< 1 >(); }

	const TType &
	y() const { return PredecessorType::template StaticGet< 1 >(); }

	Point2D&
	operator=( const PredecessorType & aVec )
	{
		PredecessorType::operator=( aVec );
		return *this;
	}
	operator PredecessorType()
	{
		return *this;
	}

	operator const PredecessorType()const
	{
		return *this;
	}
};

template< typename TType >
class Point3D: public Vector< TType, 3 >
{
public:
	typedef Vector< TType, 3 > PredecessorType;

	Point3D(){}

	explicit Point3D( const PredecessorType &aVec ): PredecessorType( aVec )
	{}

	Point3D( const TType &aX, const TType &aY, const TType &aZ ): PredecessorType( aX, aY, aZ )
	{}

	TType &
	x() { return PredecessorType::template StaticGet< 0 >(); }

	const TType &
	x() const { return PredecessorType::template StaticGet< 0 >(); }

	TType &
	y() { return PredecessorType::template StaticGet< 1 >(); }

	const TType &
	y() const { return PredecessorType::template StaticGet< 1 >(); }

	TType &
	z() { return PredecessorType::template StaticGet< 2 >(); }

	const TType &
	z() const { return PredecessorType::template StaticGet< 2 >(); }

	Point3D&
	operator=( const PredecessorType & aVec )
	{
		PredecessorType::operator=( aVec );
		return *this;
	}
	operator PredecessorType()
	{
		return *this;
	}

	operator const PredecessorType()const
	{
		return *this;
	}
};

template< typename TType >
class Point4D: public Vector< TType, 4 >
{
public:
	typedef Vector< TType, 4 > PredecessorType;

	Point4D(){}

	explicit Point4D( const PredecessorType &aVec ): PredecessorType( aVec )
	{}

	Point4D( const TType &aX, const TType &aY, const TType &aZ, const TType &aW ): PredecessorType( aX, aY, aZ, aW )
	{}

	TType &
	x() { return PredecessorType::template StaticGet< 0 >(); }

	const TType &
	x() const { return PredecessorType::template StaticGet< 0 >(); }

	TType &
	y() { return PredecessorType::template StaticGet< 1 >(); }

	const TType &
	y() const { return PredecessorType::template StaticGet< 1 >(); }

	TType &
	z() { return PredecessorType::template StaticGet< 2 >(); }

	const TType &
	z() const { return PredecessorType::template StaticGet< 2 >(); }

	TType &
	w() { return PredecessorType::template StaticGet< 3 >(); }

	const TType &
	w() const { return PredecessorType::template StaticGet< 3 >(); }

	Point4D&
	operator=( const PredecessorType & aVec )
	{
		PredecessorType::operator=( aVec );
		return *this;
	}

	operator PredecessorType()
	{
		return *this;
	}

	operator const PredecessorType()const
	{
		return *this;
	}

};
template< typename TType, unsigned tDim >
struct PointTypes;

template< typename TType >
struct PointTypes< TType, 2 >
{
	typedef Point2D< TType > Type;
};
template< typename TType >
struct PointTypes< TType, 3 >
{
	typedef Point3D< TType > Type;
};
template< typename TType >
struct PointTypes< TType, 4 >
{
	typedef Point4D< TType > Type;
};

typedef Point2D< float > Point2Df;
typedef Point3D< float > Point3Df;
typedef Point4D< float > Point4Df;

typedef Point2D< double > Point2Dd;
typedef Point3D< double > Point3Dd;
typedef Point4D< double > Point4Dd;

template< typename TType, unsigned tDim >
class Line
{
public:
	typedef typename PointTypes< TType, tDim >::Type Point;
	typedef Vector< TType, tDim > Vec;

	Line()
	{}
	Line( const Vec &a, const Vec &b ): mFirst( a ), mSecond( b )
	{}

	Line( const Point &a, const Point &b ): mFirst( a ), mSecond( b )
	{}

	Point &
	firstPoint()
	{ return mFirst; }
	const Point &
	firstPoint()const
	{ return mFirst; }

	Point &
	secondPoint()
	{ return mSecond; }
	const Point &
	secondPoint()const
	{ return mSecond; }
protected:
	Point mFirst;
	Point mSecond;
};

typedef Line< float, 2 > Line2Df;
typedef Line< double, 2 > Line2Dd;

typedef Line< float, 3 > Line3Df;
typedef Line< double, 3 > Line3Dd;

}//M4D

#endif /*GEOMETRIC_PRIMITIVES_H*/

