#ifndef DRAWING_TOOLS_H
#define DRAWING_TOOLS_H

#include "common/Common.h"
#include "common/GeometricPrimitives.h"
namespace M4D
{
	
struct BoundingBox3D
{
	static const unsigned VertexCount = 8;
	BoundingBox3D( const Vector3f &corner1 = Vector3f(), const Vector3f &corner2 = Vector3f() )
	{
		vertices[0] = Vector3f( corner1 );
		vertices[1] = Vector3f( corner2[0], corner1[1], corner1[2] );
		vertices[2] = Vector3f( corner2[0], corner2[1], corner1[2] );
		vertices[3] = Vector3f( corner1[0], corner2[1], corner1[2] );

		vertices[4] = Vector3f( corner1[0], corner1[1], corner2[2] );
		vertices[5] = Vector3f( corner2[0], corner1[1], corner2[2] );
		vertices[6] = Vector3f( corner2 );
		vertices[7] = Vector3f( corner1[0], corner2[1], corner2[2] );
	}

	Vector3f
	getMin()const
	{
		return vertices[0];
	}

	Vector3f
	getMax()const
	{
		return vertices[6];
	}

	Vector3f
	getCenter()const
	{
		return 0.5f * (vertices[0] + vertices[6]);
	}


	Vector< float, 3 >	vertices[VertexCount];
};


void
GetBBoxMinMaxDistance( 
			const BoundingBox3D		&bbox, 
			const Vector< float, 3 > 	&eyePoint, 
			const Vector< float,3 > 	&direction, 
			float 				&min, 
			float 				&max,
		       	unsigned			&minId,	
		       	unsigned			&maxId	
			);

unsigned
GetPlaneVerticesInBoundingBox( 
		const BoundingBox3D		&bbox, 
		const Vector< float, 3 > 	&planePoint, 
		const Vector< float,3 > 	&planeNormal,
		unsigned			minId,
	       	Vector< float,3 > 		vertices[]
		);

unsigned
GetPlaneVerticesInBoundingBox( 
		const BoundingBox3D		&bbox, 
		const Planef			&plane,
		unsigned			minId,
	       	Vector< float,3 > 		vertices[]
		);

unsigned
GetPlaneVerticesInBoundingBox( 
		const BoundingBox3D		&bbox, 
		const Planef			&plane,
	       	Vector< float,3 > 		vertices[]
		);

}/*namespace M4D*/


#endif /*DRAWING_TOOLS_H*/
