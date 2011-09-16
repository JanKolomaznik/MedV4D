#include "GUI/utils/DrawingTools.h"
#include "common/Common.h"
#include "common/GeometricAlgorithms.h"

namespace M4D
{

static unsigned edgeOrder[8][12] = {
		{ 10, 11,  9,  4,  8,  5,  1,  0,  6,  2,  3,  7 },
		{ 11,  8, 10,  5,  9,  6,  2,  1,  7,  3,  0,  4 },
		{  8,  9, 11,  6, 10,  7,  3,  2,  4,  0,  1,  5 },
		{  9, 10,  8,  7, 11,  4,  0,  3,  5,  1,  2,  6 },
		{  1,  0,  2,  4,  3,  7, 10, 11,  6,  9,  8,  5 },
		{  2,  1,  3,  5,  0,  4, 11,  8,  7, 10,  9,  6 },
		{  3,  2,  0,  6,  1,  5,  8,  9,  4, 11, 10,  7 },
		{  0,  3,  1,  7,  2,  6,  9, 10,  5,  8, 11,  4 }
	};


static unsigned
GetBBoxEdgePointA( unsigned idx )
{
	ASSERT( idx < 12 ); //only 12 edges

	if( idx < 8 ) {
		return idx % 4;
	}
	return idx - 4;
}

static unsigned
GetBBoxEdgePointB( unsigned idx )
{
	ASSERT( idx < 12 ); //only 12 edges

	if( idx < 4 ) {
		return (idx + 1) % 4;
	}
	if( idx < 8 ) {
		return idx;
	}
	if( idx < 11 ) {
		return idx - 3;
	}
	return 4;
}

void
GetBBoxMinMaxDistance( 
			const BoundingBox3D		&bbox, 
			const Vector< float, 3 > 	&eyePoint, 
			const Vector< float,3 > 	&direction, 
			float 				&min, 
			float 				&max, 
		       	unsigned			&minId,	
		       	unsigned			&maxId	
			)
{
	min = VectorSize( VectorProjection( direction, bbox.vertices[0] - eyePoint ) );
	minId = 0;
	max = VectorSize( VectorProjection( direction, bbox.vertices[0] - eyePoint ) );
	maxId = 0;
	for( unsigned i=1; i<8; ++i ) {
		float tmpSize = VectorSize( VectorProjection( direction, bbox.vertices[i] - eyePoint ) );
		if( tmpSize < min ) {
			min = tmpSize;
			minId = i;
		}
		if( tmpSize > max ) {
			max = tmpSize;
			maxId = i;
		}
	}

}

unsigned
GetPlaneVerticesInBoundingBox( 
		const BoundingBox3D		&bbox, 
		const Vector< float, 3 > 	&planePoint, 
		const Vector< float,3 > 	&planeNormal,
		unsigned			minId,
	       	Vector< float,3 > 		vertices[]
		)
{
	Vector< float, 3 > center;
	unsigned idx = 0;
	for( unsigned i = 0; i < 12; ++i ) {
		unsigned lineAIdx = GetBBoxEdgePointA( edgeOrder[minId][i] ); 
		unsigned lineBIdx = GetBBoxEdgePointB( edgeOrder[minId][i] );
		if( ie_UNIQUE_INTERSECTION == 
				LinePlaneIntersection( bbox.vertices[ lineAIdx ], bbox.vertices[ lineBIdx ], planePoint, planeNormal, vertices[idx] ) 
		  ) {
			++idx;
			center += vertices[idx];
			if( idx == 6 ) break;
		}
	}
	ASSERT( idx <= 6 ) //plane and box edges can have 6 intersections maximally
	return idx;
}

unsigned
GetPlaneVerticesInBoundingBox( 
		const BoundingBox3D		&bbox, 
		const Planef			&plane,
		unsigned			minId,
	       	Vector< float,3 > 		vertices[]
		)
{
	return GetPlaneVerticesInBoundingBox(
			bbox,
			plane.point(),
			plane.normal(),
			minId,
			vertices
			);

}

unsigned
GetPlaneVerticesInBoundingBox( 
		const BoundingBox3D		&bbox, 
		const Planef			&plane,
	       	Vector< float,3 > 		vertices[]
		)
{
	unsigned maxId = 0;
	Vector< float,3 > vec = VectorProjection( plane.normal(), bbox.vertices[0] - plane.point() );
	float maxSize = VectorSize( vec );
	int multiplier = sgn( vec * plane.normal() );
	if( multiplier == 0 ) { multiplier = 1; }
	for( unsigned i=1; i<8; ++i ) {
		vec = VectorProjection( plane.normal(), bbox.vertices[i] - plane.point() );
		if ( static_cast<float>( multiplier ) * (vec * plane.normal()) > 0.0f ) {
			float tmpSize = VectorSize( vec );
			if( tmpSize > maxSize ) {
				maxSize = tmpSize;
				maxId = i;
			}
		}
	}
	ASSERT( maxId < 8 );
	return GetPlaneVerticesInBoundingBox(
			bbox,
			plane.point(),
			plane.normal(),
			maxId,
			vertices
			);

}

}/*namespace M4D*/
