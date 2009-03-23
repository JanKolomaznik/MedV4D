#ifndef GEOMETRICAL_OBJECT_H
#define GEOMETRICAL_OBJECT_H

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file GeometricalObject.h 
 * @{ 
 **/

#include "common/Vector.h"

namespace Imaging
{
namespace Geometry
{

class AGeometricalObject
{
public:
	virtual ~AGeometricalObject(){}
};

template< unsigned Dim >
class AGeometricalObjectDim: public AGeometricalObject
{
public:
	static const unsigned Dimension = Dim;
};

template< typename CoordType, unsigned Dim >
class AGeometricalObjectDimPrec: public AGeometricalObjectDim< Dim >
{
public:
	typedef CoordType			Type;
	typedef Vector< Type, Dim > 	PointType;

	virtual void
	Move( PointType t ) = 0;

	virtual void
	Scale( Vector< float32, Dim > factors, PointType center ) = 0;
	
};


template < typename VectorType >
struct MoveFunctor
{
	MoveFunctor( VectorType pt ): t( pt ) {}

	void
	operator()( VectorType &v )const
		{
			v = v + t;
		}
	VectorType t;
};

template < typename VectorType >
struct ScaleFunctor
{
	typedef Vector<float32, VectorType::Dimension> ScaleFactor;

	ScaleFunctor( ScaleFactor pfactor, VectorType pcenter ): factor( pfactor ), center( pcenter ) {}

	void
	operator()( VectorType &v )const
		{
			VectorType pom = (v-center);
			for( unsigned i=0; i<VectorType::Dimension; ++i ){
				pom[i] = pom[i] * factor[i];
			}
			v = pom + center;
		}
	ScaleFactor factor;
	VectorType center;
};

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*GEOMETRICAL_OBJECT_H*/
