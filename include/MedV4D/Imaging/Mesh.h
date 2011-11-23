#ifndef MESH_H
#define MESH_H

#include <vector>

#include "MedV4D/Imaging/VertexInfo.h"
#include "MedV4D/Imaging/FaceInfo.h"
#include "MedV4D/Imaging/PointSet.h"
#include <vector>

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Mesh.h 
 * @{ 
 **/

namespace Imaging
{
namespace Geometry
{



template< typename VertexInfo = SimpleVertex_f32, typename FaceInfo = SimpleTriangle >
class Mesh: public PointSet< VertexInfo, typename VertexInfo::PositionType > 
{
public:
	typedef VertexInfo				VertexType;
	typedef typename VertexInfo::PositionType	VectorType;
	typedef FaceInfo				FaceType;
	typedef Mesh					Self;
	typedef std::vector< FaceType >			FaceList;
	typedef PointSet< VertexInfo, 
		typename VertexInfo::PositionType >	PredecessorType;
	typedef typename PredecessorType::PointList	VertexList;

	VertexList &
	GetVertices(){ return this->_points; }

	const VertexList &
	GetVertices()const{ return this->_points; }

	FaceList &
	GetFaces(){ return _faces; }

	const FaceList &
	GetFaces()const{ return _faces; }

	void
	ReserveVertices( size_t size ) { this->_points.reserve( size ); }

	void
	ReserveFaces( size_t size ) { _faces.reserve( size ); }

	size_t
	AddVertex( const VertexType & vertex ) { return this->_points.push_back( vertex ), this->_points.size()-1; }

	size_t
	AddFace( const FaceType & face ) { return _faces.push_back( face ), _faces.size()-1; }

	void
	UpdateBoundingBox(){ _THROW_ ErrorHandling::ETODO(); }
	
	void
	GetBoundingBox( VectorType &firstCorner, VectorType &secondCorner )const{ _THROW_ ErrorHandling::ETODO(); }
protected:
	RGBA_float	_color;

	FaceList	_faces;
};

typedef Mesh< SimpleVertex_f32, SimpleTriangle > SimpleMesh;

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*MESH_H*/
