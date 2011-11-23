#ifndef GEOMETRICAL_OBJECT_H
#define GEOMETRICAL_OBJECT_H

#include "MedV4D/Imaging/GeometricalObjectTypeOperations.h"
#include "MedV4D/Imaging/VertexInfo.h"
#include "MedV4D/Common/Common.h"
#include <boost/shared_ptr.hpp>

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file GeometricalObject.h 
 * @{ 
 **/


namespace Imaging
{
namespace Geometry
{


class AGeometricalObject
{
public:
	static GeometryTypeID
	GetGeometryObjectTypeID()
	{
		return GTID_AGEOMETRICAL_OBJECT;
	}

	typedef	boost::shared_ptr< AGeometricalObject >	Ptr;
	
	virtual 
	~AGeometricalObject(){}

	virtual void
	UpdateBoundingBox() = 0;
};

/*template<>
GeometryTypeID
GetGeometryObjectTypeID< AGeometricalObject >()
{
	return GTID_AGEOMETRICAL_OBJECT;
}*/

template< unsigned Dim >
class AGeometricalObjectDim: public AGeometricalObject
{
public:
	typedef	boost::shared_ptr< AGeometricalObjectDim< Dim > >	Ptr;
	static const unsigned Dimension = Dim;
};

template< typename VectorType >
class AGeometricalObjectDimPrec: public AGeometricalObjectDim< VectorType::Dimension >
{
public:
	typedef	boost::shared_ptr< AGeometricalObjectDimPrec< VectorType > >	Ptr;

	typedef typename VectorType::CoordinateType	Type;
	static const unsigned 				Dimension = VectorType::Dimension;
	//typedef VectorType		 		PointType;

	virtual void
	Move( VectorType t ) = 0;

	virtual void
	Scale( Vector< float32, Dimension > factors, VectorType center ) = 0;
	
	virtual void
	GetBoundingBox( VectorType &firstCorner, VectorType &secondCorner )const = 0;

};

template< typename VectorType >
void
TranslateVertex( VectorType &v, const VectorType t )
{
	v += t;
}

template< typename VectorType >
void
ScaleVertex( VectorType &v, const Vector<float32, VectorType::Dimension> &factor, const VectorType &center )
{
	VectorType pom = (v-center);
	for( unsigned i=0; i<VectorType::Dimension; ++i ){
		pom[i] = pom[i] * factor[i];
	}
	v = pom + center;
}

template < typename VectorType >
struct MoveFunctor
{
	MoveFunctor( VectorType pt ): t( pt ) {}


	template< typename Vertex >
	void
	operator()( Vertex &v )const
		{
			TranslateVertex( v, t );
		}
	VectorType t;
};

template < typename VectorType >
struct ScaleFunctor
{
	typedef Vector<float32, VectorType::Dimension> ScaleFactor;

	ScaleFunctor( ScaleFactor pfactor, VectorType pcenter ): factor( pfactor ), center( pcenter ) {}

	template< typename Vertex >
	void
	operator()( Vertex &v )const
		{
			ScaleVertex( v, factor, center );
		}
	ScaleFactor factor;
	VectorType center;
};

template< typename GeomObjType >
void SerializeGeometryObject( M4D::IO::OutStream &stream, const GeomObjType &obj )
{
		_THROW_ M4D::ErrorHandling::ETODO( 
				TO_STRING( "Function specialization for type \"" << 
					typeid( GeomObjType ).name() << "\" unknown." ) 
				);
}

template< typename GeomObjType >
struct SerializeGeometryObjectFtor
{
	SerializeGeometryObjectFtor( M4D::IO::OutStream &stream ): _stream( stream ) {}

	void
	operator()( const GeomObjType &obj ) {
		SerializeGeometryObject( _stream, obj );
	}

	void
	operator()( typename GeomObjType::Ptr ptr ) {
		operator()( *ptr );
	}

	M4D::IO::OutStream 	&_stream;
};


}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*GEOMETRICAL_OBJECT_H*/
