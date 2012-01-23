#include "MedV4D/GUI/utils/DrawingTools.h"
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/GeometricAlgorithms.h"
#include "MedV4D/GUI/utils/Camera.h"


namespace M4D
{

static const unsigned edgeOrder[8][12] = {
		{ 10, 11,  9,  4,  8,  5,  1,  0,  6,  2,  3,  7 },
		{ 11,  8, 10,  5,  9,  6,  2,  1,  7,  3,  0,  4 },
		{  8,  9, 11,  6, 10,  7,  3,  2,  4,  0,  1,  5 },
		{  9, 10,  8,  7, 11,  4,  0,  3,  5,  1,  2,  6 },
		{  1,  0,  2,  4,  3,  7, 10, 11,  6,  9,  8,  5 },
		{  2,  1,  3,  5,  0,  4, 11,  8,  7, 10,  9,  6 },
		{  3,  2,  0,  6,  1,  5,  8,  9,  4, 11, 10,  7 },
		{  0,  3,  1,  7,  2,  6,  9, 10,  5,  8, 11,  4 }
	};

static const unsigned edgeVertexAMapping[12] = { 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7 };
static const unsigned edgeVertexBMapping[12] = { 1, 2, 3, 0, 4, 5, 6, 7, 5, 6, 7, 4 };


static unsigned
GetBBoxEdgePointA( unsigned idx )
{
	ASSERT( idx < 12 ); //only 12 edges

	/*if( idx < 8 ) {
		return idx % 4;
	}
	return idx - 4;*/
	return edgeVertexAMapping[idx]; 
}

static unsigned
GetBBoxEdgePointB( unsigned idx )
{
	ASSERT( idx < 12 ); //only 12 edges

	/*if( idx < 4 ) {
		return (idx + 1) % 4;
	}
	if( idx < 8 ) {
		return idx;
	}
	if( idx < 11 ) {
		return idx - 3;
	}
	return 4;*/
	return edgeVertexBMapping[idx]; 
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
	//Vector< float, 3 > center;
	unsigned idx = 0;
	for( unsigned i = 0; i < 12; ++i ) {
		unsigned lineAIdx = GetBBoxEdgePointA( edgeOrder[minId][i] ); 
		unsigned lineBIdx = GetBBoxEdgePointB( edgeOrder[minId][i] );
		if( ie_UNIQUE_INTERSECTION == 
				LineSegmentPlaneIntersection( bbox.vertices[ lineAIdx ], bbox.vertices[ lineBIdx ], planePoint, planeNormal, vertices[idx] ) 
		  ) {
			++idx;
			//center += vertices[idx];
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

/*CoordType D = planeNormal * u;
	CoordType N = -planeNormal * w;

	if ( abs(D) < Epsilon ) {          // segment is parallel to plane
		if ( abs(N) < Epsilon ) {                   // segment lies in plane
		    return ie_WHOLE_INSIDE;
		} else {
		    return ie_NO_INTERSECTION;                   // no intersection
		}
	}
	// they are not parallel
	// compute intersect param
	CoordType sI = N / D;*/

struct EdgeIntersectionComputationInfo
{
	float D;
	float N0;
	Vector3f startPoint;
	Vector3f direction;
};

static inline void
fillEdgeIntersectionComputationInfo( const BoundingBox3D &bbox, const Vector3f &planeNormal, const Vector3f &planePoint, unsigned minId, EdgeIntersectionComputationInfo infos[12] )
{
	for( unsigned i = 0; i < 12; ++i ) {
		unsigned lineAIdx = GetBBoxEdgePointA( edgeOrder[minId][i] ); 
		unsigned lineBIdx = GetBBoxEdgePointB( edgeOrder[minId][i] );
		infos[i].startPoint = bbox.vertices[ lineAIdx ];
	        infos[i].direction = bbox.vertices[ lineBIdx ] - bbox.vertices[ lineAIdx ];
		infos[i].D = planeNormal * infos[i].direction;
		infos[i].N0 = -planeNormal * (bbox.vertices[ lineAIdx ] - planePoint) ;
	}
}

/*Vector< CoordType, 3 > u( lineB - lineA );
	Vector< CoordType, 3 > w( lineA - planePoint );

	//only for debugging - ensure we aren't using random location later 
	D_COMMAND( intersection = Vector< CoordType, 3 >(); );

	CoordType D = planeNormal * u;
	CoordType N = -planeNormal * w;
*/
static inline unsigned
findIntersections( float t, EdgeIntersectionComputationInfo infos[12], Vector3f vertices[] )
{
	unsigned idx = 0;
	for( unsigned i = 0; i < 12; ++i ) {
		if( abs( infos[i].D ) > Epsilon ) {
			float s = (infos[i].N0 - t)/ infos[i].D;
			if ( s > 0.0f && s < 1.0f/*intervalTest( s , 0.0f, 1.0f )*/ ) {
				vertices[idx] = infos[i].startPoint + s* infos[i].direction;
				++idx;
				if( idx == 6 ) break;
			//LOG(  infos[i].D << "; NO = " << infos[i].N0 << "; s = " << s );
			}
		}
	}
	ASSERT( idx <= 6 )
		//LOG( idx << "; t= " << t << "  ***********" );
	return idx;
}

size_t
fillPlaneBBoxIntersectionBufferFill(
		const BoundingBox3D	&bbox,
		const Camera		&camera,
		unsigned 		numberOfSteps,
		Vector3f		*vertices,
		unsigned		*indices,
		float			cutPlane,
		unsigned		primitiveRestart
		)
{
	ASSERT( vertices );
	ASSERT( indices );

	float 				min = 0; 
	float 				max = 0;
	unsigned			minId = 0;	
	unsigned			maxId = 0;	
	GetBBoxMinMaxDistance( 
		bbox, 
		camera.GetEyePosition(), 
		camera.GetTargetDirection(), 
		min, 
		max,
		minId,	
		maxId	
		);
	
	float stepSize = cutPlane * (max - min) / numberOfSteps;
	Vector< float, 3> planePoint = camera.GetEyePosition() + camera.GetTargetDirection() * max;
	Vector3f stepDirection = camera.GetTargetDirection();

	EdgeIntersectionComputationInfo infos[12];
	fillEdgeIntersectionComputationInfo( bbox, stepDirection, planePoint, minId, infos );

	Vector3f *currentVertexPtr = vertices;
	unsigned *currentIndexPtr = indices;
	size_t primitiveStartIndex = 0;
	size_t indicesSize = 0;
	for( unsigned i = 0; i < numberOfSteps; ++i ) {
		
		/*
		//Obtain intersection of the optical axis and the currently rendered plane
		planePoint -= stepSize*stepDirection;
		//Get n-gon as intersection of the current plane and bounding box
		unsigned count = M4D::GetPlaneVerticesInBoundingBox( 
				bbox, planePoint,stepDirection, minId, currentVertexPtr
				);*/
		unsigned count = findIntersections( static_cast<float>(i*stepSize), infos, currentVertexPtr );

		currentVertexPtr += count;
		primitiveStartIndex += count;
		for( unsigned j = 0; j < count; ++j ) {
			*(currentIndexPtr++) = primitiveStartIndex + j;
		}
		*(currentIndexPtr++) = primitiveRestart;
		indicesSize += count+1;
	}
	return indicesSize;
}



}/*namespace M4D*/
