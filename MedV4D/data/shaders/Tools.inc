#ifndef TOOLS_INC
#define TOOLS_INC

#include "DataStructures.inc"

#define DELTA (0.01)
#define THRESHOLD (0.1)
#define EPSILON (0.0000001f)

uint2
GetBBoxEdgePoints( uint idx )
{
	if( idx < 4 ) {
		return uint2( idx, (idx + 1) % 4 );
	}

	if( idx < 8 ) {
		return uint2( idx % 4, idx );
	}
	if( idx < 11 ) {
		return uint2( idx - 4, idx - 3 );
	}
	return uint2( idx - 4, 4 );
}

struct IntersectionResult
{
	uint resultType;
	float3 intersection;
};

bool
pointPlaneRelativePos( Plane aPlane, float3 aPoint )
{
	float3 tmp = aPoint - aPlane.point;
	return dot( tmp, aPlane.normal ) > 0.0;
}

#define CUT_PLANE_TEST( POINT )\
	if ( gEnableCutPlane && pointPlaneRelativePos( gCutPlane, POINT ) > 0.0f ) discard;

IntersectionResult
LinePlaneIntersection( 
		float3	lineA, 
		float3	lineB,
		float3	planePoint, 
		float3	planeNormal
		)
{

	float3 u = float3( lineB - lineA );
	float3 w = float3( lineA - planePoint );
	IntersectionResult result;
	result.resultType = 0;
	result.intersection = float3(0.0f, 0.0f, 0.0f);
	
	float D = dot( planeNormal, u );
	float N = dot(-planeNormal, w );

	if ( abs(D) < EPSILON ) {          // segment is parallel to plane
		result.resultType = 1;
		/*if ( abs(N) < EPSILON ) {                   // segment lies in plane
		    result.resultType = 1;
		} else {
		    result.resultType = 2;                   // no intersection
		}*/
		return result;
	}
	// they are not parallel
	// compute intersect param
	float sI = N / D;
	if (sI < 0.0f || sI > 1.0f) {
		result.resultType = 2;
		return result;                       // no intersection
	}

	result.intersection = lineA + sI * u;                 // compute segment intersect point
	return result;
}

float3
texCoordsFromPosition( float3 pos, ImageData3D image )
{
	float3 relPos = pos - image.realMinimum;

	float3 texCoord =  float3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
	if (gEnableInterpolation) {
		return texCoord;
	}
	float3 elementSize = 1.0f / image.size;
	return 0.5f * elementSize + floor( texCoord / elementSize ) * elementSize;
}

float3
texCoordsFromPositionNoInterpolation( float3 pos, ImageData3D image )
{
	float3 elementSize = 1.0f / image.size;
	float3 relPos = pos - image.realMinimum;

	float3 texCoord = float3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
	return 0.5f * elementSize + floor( texCoord / elementSize ) * elementSize;
}


float3
PreventInterpolation( float3 aTexCoord, int3 aImageDataResolution3D )
{
	float3 elementSize = 1.0f / aImageDataResolution3D;
	
	float3 coord = round( aTexCoord / elementSize ) * elementSize;
	return coord + 0.5f * elementSize;
	//return aTexCoord;
}

float3 BlinnPhongShading( float3 N, float3 V, float3 L, Material material, Light light )
{
	//half way vector
	float3 H = normalize( L + V );

	//compute ambient term
	float3 ambient = material.Ka * light.ambient;

	//compute diffuse term
	/*float diffuseLight = max( dot( L, N ), 0 );
	float3 diffuse = material.Kd * light.color * diffuseLight;

	//compute specular term
	float specularLight = pow( max( dot( H, N ), 0), material.shininess );
	if( diffuseLight <= 0.0 )
		specularLight = 0.0;
	float3 specular = material.Ks * light.color * specularLight;*/

	float4 koef = lit(dot(N,L), dot(N, H), material.shininess);
	float3 diffuse = material.Kd * light.color * koef.y;
	float3 specular = material.Ks * light.color * koef.z;

	return ambient + diffuse + specular;
}

float3 SymmetricDifference( sampler3D aTextureData, float3 aTexCoord, float3 aStep )
{
	float3 sample1, sample2;
	sample1.x = tex3D( aTextureData, coordinates - half3( aStep.x, 0.0, 0.0 ) ).x;
	sample2.x = tex3D( aTextureData, coordinates + half3( aStep.x, 0.0, 0.0 ) ).x;
	sample1.y = tex3D( aTextureData, coordinates - half3( 0.0, aStep.y, 0.0 ) ).x;
	sample2.y = tex3D( aTextureData, coordinates + half3( 0.0, aStep.y, 0.0 ) ).x;
	sample1.z = tex3D( aTextureData, coordinates - half3( 0.0, 0.0, aStep.z ) ).x;
	sample2.z = tex3D( aTextureData, coordinates + half3( 0.0, 0.0, aStep.z ) ).x;
	return sample1 - sample2;
}

#define COMPUTE_GRADIENT2( GRADIENT, CENTRAL_VALUE, TEX_DATA, COORDINATES )\
	GRADIENT.x = tex3D( TEX_DATA, COORDINATES - half3( DELTA, 0.0, 0.0 ) ).x;\
	GRADIENT.y = tex3D( TEX_DATA, COORDINATES - half3( 0.0, DELTA, 0.0 ) ).x;\
	GRADIENT.z = tex3D( TEX_DATA, COORDINATES - half3( 0.0, 0.0, DELTA ) ).x;\
	GRADIENT -= CENTRAL_VALUE.xxx

#define COMPUTE_GRADIENT( GRADIENT, CENTRAL_VALUE, TEX_DATA, COORDINATES )\
	GRADIENT.x = tex3D( TEX_DATA, COORDINATES - half3( DELTA, 0.0, 0.0 ) ).x - tex3D( TEX_DATA, COORDINATES + half3( DELTA, 0.0, 0.0 ) ).x;\
	GRADIENT.y = tex3D( TEX_DATA, COORDINATES - half3( 0.0, DELTA, 0.0 ) ).x - tex3D( TEX_DATA, COORDINATES + half3( 0.0, DELTA, 0.0 ) ).x;\
	GRADIENT.z = tex3D( TEX_DATA, COORDINATES - half3( 0.0, 0.0, DELTA ) ).x - tex3D( TEX_DATA, COORDINATES + half3( 0.0, 0.0, DELTA ) ).x;

#define SYMMETRIC_DIFFERENCE( SAMPLE1, SAMPLE2, TEX_DATA, COORDINATES )\
	SAMPLE1.x = tex3D( TEX_DATA, COORDINATES - half3( DELTA, 0.0, 0.0 ) ).x;\
	SAMPLE2.x = tex3D( TEX_DATA, COORDINATES + half3( DELTA, 0.0, 0.0 ) ).x;\
	SAMPLE1.y = tex3D( TEX_DATA, COORDINATES - half3( 0.0, DELTA, 0.0 ) ).x;\
	SAMPLE2.y = tex3D( TEX_DATA, COORDINATES + half3( 0.0, DELTA, 0.0 ) ).x;\
	SAMPLE1.z = tex3D( TEX_DATA, COORDINATES - half3( 0.0, 0.0, DELTA ) ).x;\
	SAMPLE2.z = tex3D( TEX_DATA, COORDINATES + half3( 0.0, 0.0, DELTA ) ).x;

#define DIFFERENCE( SAMPLE, TEX_DATA, COORDINATES )\
	SAMPLE.x = tex3D( TEX_DATA, COORDINATES - half3( DELTA, 0.0, 0.0 ) ).x;\
	SAMPLE.y = tex3D( TEX_DATA, COORDINATES - half3( 0.0, DELTA, 0.0 ) ).x;\
	SAMPLE.z = tex3D( TEX_DATA, COORDINATES - half3( 0.0, 0.0, DELTA ) ).x;\


float3 Difference( sampler3D aTextureData, float3 aTexCoord, float3 aStep, float aValue )
{
	float3 sample1;
	sample1.x = tex3D( aTextureData, coordinates - half3( aStep.x, 0.0, 0.0 ) ).x;
	sample1.y = tex3D( aTextureData, coordinates - half3( 0.0, aStep.y, 0.0 ) ).x;
	sample1.z = tex3D( aTextureData, coordinates - half3( 0.0, 0.0, aStep.z ) ).x;
	return sample1 - aValue.xxx;
}


float4
ApplyTransferFunction1Dbac( float value, sampler1D aTransferFunction1D, float2 interval )
{
	float range = interval[1] - interval[0];
	float remappedValue = ( value - interval[0] ) / range;

	return tex1D( aTransferFunction1D, remappedValue );
}

float4
ApplyTransferFunction1D( float value, TransferFunction1D aTransferFunction1D )
{
	float range = aTransferFunction1D.interval[1] - aTransferFunction1D.interval[0];
	float remappedValue = ( value - aTransferFunction1D.interval[0] ) / range;

	return tex1D( aTransferFunction1D.data, remappedValue );
}

#endif /*TOOLS_INC*/
