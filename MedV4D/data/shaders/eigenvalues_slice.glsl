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
	
	const float minimum = -5;
	const float maximum = -2;
	
	vec3 unmappedValues = (mappedValues * maximum) + minimum;
	
	float lowBand = wlWindow.y - (wlWindow.x * 0.5f);
  float highBand = wlWindow.y + (wlWindow.x * 0.5f);
  float multiplier = wlWindow.z;
	
	float value = computeVesselness(mappedValues.x, mappedValues.y, mappedValues.z);
	fragmentColor = vec4(value, 0, 0, 1);
	
	/*if (mappedValues.x == -100)
		fragmentColor = vec4(0, 1, 0, 1);
	else if (mappedValues.x == 0.5)
		fragmentColor = vec4(1, 0, 0, 1);
	else
		fragmentColor = vec4(0, 0, 0, 0);*/
	
	/*if (unmappedValues.x < -1.5 && unmappedValues.x > -2.5)
		fragmentColor = vec4(0, 1, 0, 1);
	else if (unmappedValues.x < -50000)
		fragmentColor = vec4(1, 0, 1, 1);
	else
		fragmentColor = vec4(0, 0, 0, 0);*/
	
  /*float lowBand = wlWindow.y - (wlWindow.x * 0.5f);
  float highBand = wlWindow.y + (wlWindow.x * 0.5f);
  float multiplier = wlWindow.z;
				
  vec3 eigenvalues = decodeEigenvalues(encodedEigenvalues);
  float value = (computeVesselness(eigenvalues.x, eigenvalues.y, eigenvalues.z) - lowBand);// * multiplier;
	
	//fragmentColor = vec4(1, 1, 1, value);
	fragmentColor = vec4(encodedEigenvalues * multiplier, 1);*/
}
