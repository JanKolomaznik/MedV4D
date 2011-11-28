#ifndef GEOMETRIC_PRIMITIVES_H
#define GEOMETRIC_PRIMITIVES_H

#include "MedV4D/Common/Vector.h"

namespace M4D
{

template< typename TType >
class Point2D: public Vector< TType, 2 >
{
public:
	typedef Vector< TType, 2 > PredecessorType;

	Point2D(){}

	template< typename TCType  >
	explicit Point2D( const Vector< TCType, 2 > &aVec ): PredecessorType( aVec )
	{}

	Point2D( const TType &aX, const TType &aY ): PredecessorType( aX, aY )
	{}

	TType &
	x() { return PredecessorType::template StaticGet< 0 >(); }

	TType
	x() const { return PredecessorType::template StaticGet< 0 >(); }

	TType &
	y() { return PredecessorType::template StaticGet< 1 >(); }

	TType 
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

	template< typename TCType  >
	explicit Point3D( const Vector< TCType, 3 > &aVec ): PredecessorType( aVec )
	{}

	Point3D( const TType &aX, const TType &aY, const TType &aZ ): PredecessorType( aX, aY, aZ )
	{}

	TType &
	x() { return PredecessorType::template StaticGet< 0 >(); }

	TType
	x() const { return PredecessorType::template StaticGet< 0 >(); }

	TType &
	y() { return PredecessorType::template StaticGet< 1 >(); }

	TType
	y() const { return PredecessorType::template StaticGet< 1 >(); }

	TType &
	z() { return PredecessorType::template StaticGet< 2 >(); }

	TType
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

	template< typename TCType  >
	explicit Point4D( const Vector< TCType, 4 > &aVec ): PredecessorType( aVec )
	{}

	Point4D( const TType &aX, const TType &aY, const TType &aZ, const TType &aW ): PredecessorType( aX, aY, aZ, aW )
	{}

	TType &
	x() { return PredecessorType::template StaticGet< 0 >(); }

	TType
	x() const { return PredecessorType::template StaticGet< 0 >(); }

	TType &
	y() { return PredecessorType::template StaticGet< 1 >(); }

	TType
	y() const { return PredecessorType::template StaticGet< 1 >(); }

	TType &
	z() { return PredecessorType::template StaticGet< 2 >(); }

	TType 
	z() const { return PredecessorType::template StaticGet< 2 >(); }

	TType &
	w() { return PredecessorType::template StaticGet< 3 >(); }

	TType 
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

template< typename TType >
class Plane
{
public:
	typedef Vector< TType, 3 > Vec;
	typedef typename PointTypes< TType, 3 >::Type Point;

	Plane()
	{}
	Plane( const Point &aPoint, const Vec &aNormal ): mPoint( aPoint ), mNormal( aNormal )
	{}

	Plane( const Vec &aPoint, const Vec &aNormal ): mPoint( aPoint ), mNormal( aNormal )
	{}

	Point&
	point()
	{ return mPoint; }

	const Point&
	point()const
	{ return mPoint; }

	Vec&
	normal()
	{ return mNormal; }

	const Vec&
	normal()const
	{ return mNormal; }
protected:
	Point mPoint;
	Vec mNormal;
};

typedef Plane< float > Planef;
typedef Plane< double > Planed;

}//M4D

#endif /*GEOMETRIC_PRIMITIVES_H*/

