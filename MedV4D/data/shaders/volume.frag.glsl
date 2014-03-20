#version 150
//precision highp float;

struct ImageData3D
{
	sampler3D data;
	ivec3 size;
	vec3 realSize;
	vec3 realMinimum;
	vec3 realMaximum;
};

struct TransferFunction1D
{
	sampler1D data;
	vec2 interval;
	int sampleCount;
};

uniform ImageData3D gPrimaryImageData3D;
//uniform ImageData3D gSecondaryImageData3D;

//int3 gImageDataResolution3D = {0, 0, 0};
uniform vec2 gMappedIntervalBands;

//Light gLight;
//Camera gCamera;

uniform TransferFunction1D gTransferFunction1D;
uniform float gRenderingSliceThickness;

vec4
applyTransferFunction1D(float value, TransferFunction1D aTransferFunction1D)
{
	float range = aTransferFunction1D.interval[1] - aTransferFunction1D.interval[0];
	float remappedValue = ( value - aTransferFunction1D.interval[0] ) / range;

	return texture( aTransferFunction1D.data, remappedValue );
} 

vec3
texCoordsFromPosition(vec3 pos, ImageData3D image)
{
	vec3 relPos = pos - image.realMinimum;

	vec3 texCoord =  vec3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
	//if (gEnableInterpolation) {
	//	return texCoord;
	//}
	vec3 elementSize = 1.0f / image.size;
	return 0.5f * elementSize + floor( texCoord / elementSize ) * elementSize;
}

vec3
texCoordsFromPositionNoInterpolation(vec3 pos, ImageData3D image)
{
	vec3 elementSize = 1.0f / image.size;
	vec3 relPos = pos - image.realMinimum;

	vec3 texCoord = vec3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
	return 0.5f * elementSize + floor( texCoord / elementSize ) * elementSize;
}
 

vec4 
transferFunction1D( 
		vec3 aPosition, 
		ImageData3D aTextureData,
		TransferFunction1D aTransferFunction1D,
		float aRenderingSliceThickness,
		vec2 aMappedIntervalBands)
{
	//CUT_PLANE_TEST( aPosition );
	
	vec3 coordinates = texCoordsFromPosition( aPosition, aTextureData );
	//return vec4(1.0, 0.0, 0.0, 0.1);
	//return vec4(coordinates, 0.2f);
	//return texture(aTextureData.data, coordinates) * 1000;
	float range = aMappedIntervalBands[1] - aMappedIntervalBands[0];
	float value = (texture(aTextureData.data, coordinates).x * range) + aMappedIntervalBands[0];
		
	vec4 outputColor = applyTransferFunction1D( value, aTransferFunction1D );
	outputColor.a = 1.0f - pow(1.0f - outputColor.a, aRenderingSliceThickness);

	return outputColor;
}

in vec3 positionInImage;
out vec4 fragmentColor;

void main(void)
{
    //vec3 eyeSpaceLigthDirection = vec3(0.0,-1.0,10.0);
    //float diffuse = max(0.0,dot(normalize(n),eyeSpaceLigthDirection));
	fragmentColor = vec4(1.0, 0.0, 0.0, 0.1);
	/*if (gPrimaryImageData3D.realSize.x / 2 > positionInImage.x) {
		fragmentColor = vec4(1.0, 0.0, 1.0, 0.5);
	}*/
	fragmentColor = transferFunction1D(
						positionInImage,
						gPrimaryImageData3D,
						gTransferFunction1D,
						gRenderingSliceThickness,
						gMappedIntervalBands);
	//fragmentColor = positionInImage / 300;
	//fragmentColor.a = max(0.5, fragmentColor.a);
}