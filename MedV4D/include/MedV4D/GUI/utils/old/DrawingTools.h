#ifndef DRAWING_TOOLS_H
#define DRAWING_TOOLS_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/GeometricPrimitives.h"

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <soglu/Primitives.hpp>
#include <soglu/BoundingBox.hpp>

class Camera;
namespace M4D
{
	
struct BoundingBox3D
{
	static const unsigned VertexCount = 8;
	BoundingBox3D( const glm::fvec3 &corner1 = glm::fvec3(), const glm::fvec3 &corner2 = glm::fvec3() )
	{
		vertices[0] = glm::fvec3( corner1 );
		vertices[1] = glm::fvec3( corner2.x, corner1.y, corner1.z );
		vertices[2] = glm::fvec3( corner2.x, corner2.y, corner1.z );
		vertices[3] = glm::fvec3( corner1.x, corner2.y, corner1.z );

		vertices[4] = glm::fvec3( corner1.x, corner1.y, corner2.z );
		vertices[5] = glm::fvec3( corner2.x, corner1.y, corner2.z );
		vertices[6] = glm::fvec3( corner2 );
		vertices[7] = glm::fvec3( corner1.x, corner2.y, corner2.z );
	}

	glm::fvec3
	getMin()const
	{
		return vertices[0];
	}

	glm::fvec3
	getMax()const
	{
		return vertices[6];
	}

	glm::fvec3
	getCenter()const
	{
		return 0.5f * (vertices[0] + vertices[6]);
	}

	glm::fvec3	vertices[VertexCount];
};

inline std::ostream &
operator<<( std::ostream & s, const BoundingBox3D &bbox )
{
	s << "BBox3d[ " << bbox.getMin() << "; "  << bbox.getMin() << " ]";
	return s;
}

void
getBBoxMinMaxDistance( 
			const BoundingBox3D	&bbox, 
			const glm::fvec3 	&eyePoint, 
			const glm::fvec3  	&direction, 
			float 			&min, 
			float 			&max,
		       	unsigned		&minId,
		       	unsigned		&maxId
			);

unsigned
getPlaneVerticesInBoundingBox( 
		const BoundingBox3D	&bbox, 
		const glm::fvec3 	&planePoint, 
		const glm::fvec3 	&planeNormal,
		unsigned			minId,
	       	glm::fvec3 		vertices[]
		);


unsigned
GetPlaneVerticesInBoundingBox( 
		const soglu::BoundingBox3D		&bbox, 
		const glm::fvec3 	&planePoint, 
		const glm::fvec3 	&planeNormal,
		unsigned			minId,
	       	glm::fvec3 		vertices[]
		);

unsigned
GetPlaneVerticesInBoundingBox( 
		const soglu::BoundingBox3D		&bbox, 
		const soglu::Planef			&plane,
		unsigned			minId,
	       	glm::fvec3 		vertices[]
		);

unsigned
GetPlaneVerticesInBoundingBox( 
		const soglu::BoundingBox3D		&bbox, 
		const soglu::Planef			&plane,
	       	glm::fvec3 		vertices[]
		);

size_t
fillPlaneBBoxIntersectionBufferFill(
		const BoundingBox3D	&bbox,
		const Camera		&camera,
		unsigned 		numberOfSteps,
		Vector3f		*vertices,
		unsigned		*indices,
		float			cutPlane,
		unsigned		primitiveRestart
		);


}/*namespace M4D*/


#endif /*DRAWING_TOOLS_H*/
