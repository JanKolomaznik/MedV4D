uniform sampler2D gDepthBuffer;
uniform sampler2D gNoiseMap;
uniform float gJitterStrength = 1.0f;
uniform vec2 gNoiseMapSize;

uniform ImageData3D gPrimaryImageData3D;
//uniform ImageData3D gSecondaryImageData3D;

uniform vec2 gMappedIntervalBands;
uniform vec2 gWLWindow;

uniform Light gLight;
uniform Camera gCamera;

uniform TransferFunction1D gTransferFunction1D;
uniform float gRenderingSliceThickness;

uniform ViewSetup gViewSetup;
uniform vec2 gWindowSize;
/*vec4 
applyWLWindow2( 
		vec3 aPosition, 
		ImageData3D aTextureData, 
		vec3 aWLWindow
		)
{
	float lowBand = aWLWindow.y - (aWLWindow.x * 0.5f);
	float highBand = aWLWindow.y + (aWLWindow.x * 0.5f);
	float multiplier = aWLWindow.z;

	vec3 coordinates = texCoordsFromPosition( aPosition, aTextureData );
	float value = clamp(
			(texture(aTextureData.data, coordinates).x - lowBand) * multiplier,
			0.0f,
			1.0f
			);
		
	return vec4( 0.9f, 0.9f, 0.9f, 0.5f * value);//(value, value, value, 1.0f );
}*/

float 
applyWLWindow2( 
		vec3 aPosition, 
		ImageData3D aTextureData, 
		vec3 aWLWindow
		)
{
	float lowBand = aWLWindow.y - (aWLWindow.x * 0.5f);
	float highBand = aWLWindow.y + (aWLWindow.x * 0.5f);
	float multiplier = aWLWindow.z;

	vec3 coordinates = texCoordsFromPosition( aPosition, aTextureData );
	float value = clamp(
			(texture(aTextureData.data, coordinates).x - lowBand) * multiplier,
			0.0f,
			1.0f
			);
		
	return value;
}

in vec3 positionInImage;
out vec4 fragmentColor;

//#define ENABLE_PREINTEGRATED_TRANSFER_FUNCTION
//#define ENABLE_JITTERING
void main(void)
{
	vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
	vec3 coordinates = positionInImage;

	vec3 dir = normalize(coordinates - gCamera.eyePosition);
	float value = -1.0;
	for (int i = 0; i < 1000; ++i) {
		vec3 point = coordinates + i * dir;

		vec4 depth_vec = gViewSetup.modelViewProj * vec4(point, 1.0);
		float currentDepth = (depth_vec.z / depth_vec.w + 1.0) * 0.5;
		float depth = texture(gDepthBuffer, gl_FragCoord.st / gWindowSize).x;
		if (depth < currentDepth) {
			break;
		}
		float value2 = applyWLWindow2(
				point,
				gPrimaryImageData3D,
				vec3( 
					gWLWindow.x / (gMappedIntervalBands[1] - gMappedIntervalBands[0]), 
					gWLWindow.y / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
					(gMappedIntervalBands[1] - gMappedIntervalBands[0]) / gWLWindow.x 
					)
				);
		value = max(value, value2);
	}
	fragmentColor = value < 0.0 ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(value, value, value, 1.0);
}
