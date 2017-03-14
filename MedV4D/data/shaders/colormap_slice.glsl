uniform ImageData3D gPrimaryImageData3D;
uniform vec2 gMappedIntervalBands;

/*vec3
texCoordsFromPosition(vec3 pos, ImageData3D image)
{
	vec3 relPos = pos - image.realMinimum;
	return vec3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
}*/

vec3
texCoordsFromPositionNoInterpolation(vec3 pos, ImageData3D image)
{
	vec3 elementSize = 1.0f / image.size;
	vec3 relPos = pos - image.realMinimum;

	vec3 texCoord = vec3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
	return 0.5f * elementSize + floor( texCoord / elementSize ) * elementSize;
}



vec4 
colorMap(vec3 aPosition, ImageData3D aTextureData)
{
	vec3 coordinates = texCoordsFromPositionNoInterpolation( aPosition, aTextureData );
	float value = 1000000*texture(aTextureData.data, coordinates).x;
	
	float r = (int(35 * (value + 132)) % 256) / 256.0f;
	float g = (int(364 * (value + 532)) % 256) / 256.0f;
	float b = (int(12 * (value + 4899)) % 256) / 256.0f;
	
	return vec4(r, g, b, 1.0);
	/*if (value > 0.3) {

		if (value > 0.8) {
			return vec4(0.0, 0.0, 1.0, 0.3);
		} else {
			return vec4(1.0, 1.0, 0.0, 0.3);
		}
	} else {
		return vec4(1.0, 1.0, 0.0, 0.0);
	}*/
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
