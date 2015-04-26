uniform ImageData3D gPrimaryImageData3D;
uniform vec2 gMappedIntervalBands;
uniform vec2 gWLWindow;

vec3
texCoordsFromPosition(vec3 pos, ImageData3D image)
{
	vec3 relPos = pos - image.realMinimum;
	return vec3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
}

vec3
applyWLWindowVector(
vec3 aPosition,
ImageData3D aTextureData,
vec3 aWLWindow
)
{
  vec3 coordinates = texCoordsFromPosition(aPosition, aTextureData);
  return texture(aTextureData.data, coordinates).rgb;
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
	/*float value = clamp(
			(texture(aTextureData.data, coordinates).x - lowBand) * multiplier,
			0.0f,
			1.0f
			);
		
	return vec4(value, value, value, 1.0f );*/

	vec3 value = clamp(
			(texture(aTextureData.data, coordinates).rgb - lowBand) * multiplier,
			0.0f,
			1.0f
			);
		
	return vec4(value.r, value.g, value.b, 1.0f );
}

in vec3 positionInImage;
out vec4 fragmentColor;

void main(void)
{
	vec3 wlWindow = vec3(
		gWLWindow.x / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
		gWLWindow.y / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
		(gMappedIntervalBands[1] - gMappedIntervalBands[0]) / gWLWindow.x
		);
	vec3 mappedValues = applyWLWindowVector(
				positionInImage,
				gPrimaryImageData3D,
				wlWindow
				);
	
	float lowBand = wlWindow.y - (wlWindow.x * 0.5f);
  float highBand = wlWindow.y + (wlWindow.x * 0.5f);
  float multiplier = wlWindow.z;
	
	vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(mappedValues.x, mappedValues.y, mappedValues.z);
	
	float displayedEigenvalue = sortedEigenvalues.z;
	
	if (displayedEigenvalue > 0)
	{
		fragmentColor = vec4(0, 0, abs(displayedEigenvalue), 1);
	}
	else
	{
		fragmentColor = vec4(abs(displayedEigenvalue), 0, 0, 1);
	}
}
