//#version 150
//precision highp float;


uniform sampler2D gNoiseMap;
uniform float gJitterStrength = 1.0f;
uniform vec2 gNoiseMapSize;

uniform ImageData3D gPrimaryImageData3D;
//uniform ImageData3D gSecondaryImageData3D;

//int3 gImageDataResolution3D = {0, 0, 0};
uniform vec2 gMappedIntervalBands;
uniform vec2 gWLWindow;

uniform Light gLight;
uniform Camera gCamera;

uniform TransferFunction1D gTransferFunction1D;
uniform float gRenderingSliceThickness;

in vec3 positionInImage;
out vec4 fragmentColor;

//#define ENABLE_PREINTEGRATED_TRANSFER_FUNCTION
//#define ENABLE_JITTERING
void main(void)
{
	vec3 coordinates = positionInImage;
#ifdef ENABLE_JITTERING
	float offset = texture(gNoiseMap, gl_FragCoord.xy / gNoiseMapSize.xy ).r * gJitterStrength * gRenderingSliceThickness;
	coordinates = coordinates + offset * gCamera.viewDirection;
#endif //ENABLE_JITTERING

#ifdef TRANSFER_FUNCTION_RENDERING
#  ifdef ENABLE_PREINTEGRATED_TRANSFER_FUNCTION
	vec4 color = preintegratedTransferFunction1D(
				coordinates,
				gPrimaryImageData3D,
				gTransferFunction1D,
				gRenderingSliceThickness,
				gMappedIntervalBands,
				gCamera.viewDirection);
#  else	
	vec4 color = transferFunction1D(
				coordinates,
				gPrimaryImageData3D,
				gTransferFunction1D,
				gMappedIntervalBands);
#  endif //ENABLE_PREINTEGRATED_TRANSFER_FUNCTION

#  ifdef ENABLE_SHADING
	color = doShading(
		coordinates,
		color,
		gPrimaryImageData3D,
		gLight,
		gCamera.eyePosition
		);
#  endif //ENABLE_SHADING
#endif //TRANSFER_FUNCTION_RENDERING

#ifdef DENSITY_RENDERING
	vec4 color = applyWLWindow(
				positionInImage,
				gPrimaryImageData3D,
				vec3( 
					gWLWindow.x / (gMappedIntervalBands[1] - gMappedIntervalBands[0]), 
					gWLWindow.y / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
					(gMappedIntervalBands[1] - gMappedIntervalBands[0]) / gWLWindow.x 
					)
				);
#endif //DENSITY_RENDERING
	color.a = 1.0f - pow(1.0f - color.a, gRenderingSliceThickness);
	fragmentColor = color;
}
