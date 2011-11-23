#ifndef _VERTEX_INFO_H
#define _VERTEX_INFO_H


#include "MedV4D/Common/Vector.h"
#include "MedV4D/Common/Color.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file VertexInfo.h 
 * @{ 
 **/

namespace Imaging
{
namespace Geometry
{


template< typename CoordType >
struct PositionPart
{
	typedef Vector< CoordType, 3 >	PositionType;

	PositionType	position;
};

template< typename CoordType >
struct NormalPart
{
	typedef Vector< CoordType, 3 >	NormalType;

	NormalType	normal;
};

template< typename CoordType >
struct NormalDummy
{
};

struct ColorPart
{
	RGB_float	color;
};

struct ColorDummy
{
};


template< 
	typename CoordType, 
	template <typename CType> class NormalPart = NormalDummy, 
	typename ColorPart = ColorDummy 
	>
struct VertexInfo: public PositionPart< CoordType >, public NormalPart< CoordType >, public ColorPart
{

};

template< 
	typename VertexType,
       	typename VectorType	
	>
void
TranslateVertex( VertexType &v, const VectorType t )
{
	v.position += t;
}

template< 
	typename VertexType,
       	typename VectorType	
	>
void
ScaleVertex( VertexType &v, const Vector<float32, VectorType::Dimension> &factor, const VectorType &center )
{
	VectorType pom = (v.position-center);
	for( unsigned i=0; i<VectorType::Dimension; ++i ){
		pom[i] = pom[i] * factor[i];
	}
	v.position = pom + center;
}

typedef VertexInfo< float32 > SimpleVertex_f32;
typedef VertexInfo< float64 > SimpleVertex_f64;


}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*_VERTEX_INFO_H*/


