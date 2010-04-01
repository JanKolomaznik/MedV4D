#ifndef DRAWING_TOOLS_H
#define DRAWING_TOOLS_H

#include "common/Common.h"
namespace M4D
{
	
struct BoundingBox3D
{
	static const unsigned VertexCount = 8;
	BoundingBox3D( const Vector< float, 3 > &corner1 = Vector< float, 3 >(), const Vector< float, 3 > &corner2 = Vector< float, 3 >() )
	{
		vertices[0] = Vector< float, 3 >( corner1 );
		vertices[1] = Vector< float, 3 >( corner2[0], corner1[1], corner1[2] );
		vertices[2] = Vector< float, 3 >( corner2[0], corner2[1], corner1[2] );
		vertices[3] = Vector< float, 3 >( corner1[0], corner2[1], corner1[2] );

		vertices[4] = Vector< float, 3 >( corner1[0], corner1[1], corner2[2] );
		vertices[5] = Vector< float, 3 >( corner2[0], corner1[1], corner2[2] );
		vertices[6] = Vector< float, 3 >( corner2 );
		vertices[7] = Vector< float, 3 >( corner1[0], corner2[1], corner2[2] );
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

}/*namespace M4D*/


#endif /*DRAWING_TOOLS_H*/
