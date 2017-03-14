uniform ImageData3D gPrimaryImageData3D;
uniform vec2 gMappedIntervalBands;

vec3
texCoordsFromPosition(vec3 pos, ImageData3D image)
{
	vec3 relPos = pos - image.realMinimum;
	return vec3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
}

vec4 
colorMap(vec3 aPosition, ImageData3D aTextureData)
{
	//return vec4(1.0, 0.0, 0.0, 1.0);
	vec3 coordinates = texCoordsFromPosition( aPosition, aTextureData );
	float value = texture(aTextureData.data, coordinates).x;
		
	if (value > 0.3) {
		if (value > 0.8) {
			return vec4(0.0, 0.0, 1.0, 0.3);
		} else {
			return vec4(1.0, 0.0, 0.0, 0.3);
		}
	}
	return vec4(0.0, 0.0, 0.0, 0.0);
}

in vec3 positionInImage;
out vec4 fragmentColor;

void main(void)
{
	fragmentColor = colorMap(
		positionInImage,
		gPrimaryImageData3D
		);
}
