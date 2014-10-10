
//precision highp float;

struct ImageData3D
{
	sampler3D data;
	ivec3 size;
	vec3 realSize;
	vec3 realMinimum;
	vec3 realMaximum;
};

uniform ImageData3D gPrimaryImageData3D;
uniform vec2 gMappedIntervalBands;
uniform vec2 gWLWindow;

vec3
texCoordsFromPosition(vec3 pos, ImageData3D image)
{
	vec3 relPos = pos - image.realMinimum;
	return vec3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
}

vec4 
applyWLWindow( 
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
		
	return vec4(value, value, value, 1.0f );
}

in vec3 positionInImage;
out vec4 fragmentColor;

void main(void)
{
	fragmentColor = vec4(0, 1, 0, 1);
	fragmentColor = applyWLWindow(
						positionInImage,
						gPrimaryImageData3D,
						vec3( 
							gWLWindow.x / (gMappedIntervalBands[1] - gMappedIntervalBands[0]), 
							gWLWindow.y / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
							(gMappedIntervalBands[1] - gMappedIntervalBands[0]) / gWLWindow.x 
							)
						);
}
