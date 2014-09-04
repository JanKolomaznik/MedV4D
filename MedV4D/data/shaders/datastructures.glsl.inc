#define THRESHOLD (0.1)
#define DELTA (0.01)
#define COMPUTE_GRADIENT( GRADIENT, CENTRAL_VALUE, TEX_DATA, COORDINATES )\
	GRADIENT.x = texture( TEX_DATA, COORDINATES - vec3( DELTA, 0.0, 0.0 ) ).x - texture( TEX_DATA, COORDINATES + vec3( DELTA, 0.0, 0.0 ) ).x;\
	GRADIENT.y = texture( TEX_DATA, COORDINATES - vec3( 0.0, DELTA, 0.0 ) ).x - texture( TEX_DATA, COORDINATES + vec3( 0.0, DELTA, 0.0 ) ).x;\
	GRADIENT.z = texture( TEX_DATA, COORDINATES - vec3( 0.0, 0.0, DELTA ) ).x - texture( TEX_DATA, COORDINATES + vec3( 0.0, 0.0, DELTA ) ).x;


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

struct TransferFunction2D
{
	sampler2D data;
	vec2 from;
	vec2 to;
	int xSamples;
	int ySamples;
};

struct Camera
{
	vec3 eyePosition;
	vec3 viewDirection;
	vec3 upDirection;	
};

struct ViewSetup
{
	mat4 modelViewProj;
	mat4 modelMatrix;
	mat4 projMatrix;
	mat4 viewMatrix;
};

struct Light
{
	vec3 color;
	vec3 ambient;
	vec3 position;
};

struct Material
{
	vec3	Ka;
	vec3	Kd;
	vec3	Ks;
	float	shininess;
};


vec4 lit(float NdotL, float NdotH, float m) 
{
  float specular = (NdotL > 0) ? pow(max(0.0, NdotH), m) : 0.0;
  return vec4(1.0, max(0.0, NdotL), specular, 1.0);
}

vec3 blinnPhongShading(vec3 N, vec3 V, vec3 L, Material material, Light light)
{
	//half way vector
	vec3 H = normalize( L + V );

	//compute ambient term
	vec3 ambient = material.Ka * light.ambient;

	vec4 koef = lit(dot(N, L), dot(N, H), material.shininess);
	vec3 diffuse = material.Kd * light.color * koef.y;
	vec3 specular = material.Ks * light.color * koef.z;

	return ambient + diffuse + specular;
}

vec4
applyTransferFunction1D(float value, TransferFunction1D aTransferFunction1D)
{
	float range = aTransferFunction1D.interval[1] - aTransferFunction1D.interval[0];
	float remappedValue = ( value - aTransferFunction1D.interval[0] ) / range;

	return texture( aTransferFunction1D.data, remappedValue );
} 

vec4
applyIntegralTransferFunction1D( 
		vec2 values,
		ImageData3D aTextureData,
		TransferFunction1D aIntegralTransferFunction1D
		)
{
	float v1 = max(values.x, values.y);
	float v2 = min(values.x, values.y);
	v2 = min(v2, v1 - 0.5f); //Prevent division by zero
	float factor = 1.0f / (v1 - v2);
	vec4 color1 = applyTransferFunction1D(v1, aIntegralTransferFunction1D);
	vec4 color2 = applyTransferFunction1D(v2, aIntegralTransferFunction1D);
	vec4 color = (color1 - color2) * factor;
	return color;
}




vec3
texCoordsFromPosition(vec3 pos, ImageData3D image)
{
	vec3 relPos = pos - image.realMinimum;
	return vec3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
}

vec3
texCoordsFromPositionNoInterpolation(vec3 pos, ImageData3D image)
{
	vec3 elementSize = 1.0f / image.size;
	vec3 relPos = pos - image.realMinimum;

	vec3 texCoord = vec3( relPos.x / image.realSize.x, relPos.y / image.realSize.y, relPos.z / image.realSize.z );
	return 0.5f * elementSize + floor( texCoord / elementSize ) * elementSize;
}

float 
getUnmappedValue(
		vec3 aPosition, 
		ImageData3D aTextureData,
		vec2 aMappedIntervalBands
		)
{
	float range = aMappedIntervalBands[1] - aMappedIntervalBands[0];
	vec3 coordinates = texCoordsFromPosition(aPosition, aTextureData);
	return (texture(aTextureData.data, coordinates).x * range) + aMappedIntervalBands[0];
}

vec3 
computeGradient(ImageData3D aImage, vec2 aMappedIntervalBands, float aCentralValue, vec3 aPosition)
{
	vec3 gradient;
	/*gradient.x = (aCentralValue - getUnmappedValue(vec3(aPosition.x - DELTA, aPosition.yz), aImage, aMappedIntervalBands)) / DELTA;
	gradient.y = (aCentralValue - getUnmappedValue(vec3(aPosition.x, aPosition.y - DELTA, aPosition.z), aImage, aMappedIntervalBands)) / DELTA;
	gradient.z = (aCentralValue - getUnmappedValue(vec3(aPosition.xy, aPosition.z - DELTA), aImage, aMappedIntervalBands)) / DELTA;*/
	gradient.x = (getUnmappedValue(vec3(aPosition.x + DELTA, aPosition.yz), aImage, aMappedIntervalBands) - getUnmappedValue(vec3(aPosition.x - DELTA, aPosition.yz), aImage, aMappedIntervalBands)) / (2*DELTA);
	gradient.y = (getUnmappedValue(vec3(aPosition.x, aPosition.y + DELTA, aPosition.z), aImage, aMappedIntervalBands) - getUnmappedValue(vec3(aPosition.x, aPosition.y - DELTA, aPosition.z), aImage, aMappedIntervalBands)) / (2*DELTA);
	gradient.z = (getUnmappedValue(vec3(aPosition.xy, aPosition.z + DELTA), aImage, aMappedIntervalBands) - getUnmappedValue(vec3(aPosition.xy, aPosition.z - DELTA), aImage, aMappedIntervalBands)) / (2*DELTA);
	return gradient;
}
 
vec4 
preintegratedTransferFunction1D( 
		vec3 aPosition, 
		ImageData3D aTextureData,
		TransferFunction1D aIntegralTransferFunction1D,
		float aRenderingSliceThickness,
		vec2 aMappedIntervalBands,
		vec3 aViewDirection)
{
	float range = aMappedIntervalBands[1] - aMappedIntervalBands[0];
	vec3 coordinates1 = texCoordsFromPosition( aPosition, aTextureData );
	vec3 coordinates2 = texCoordsFromPosition( aPosition + aRenderingSliceThickness * aViewDirection , aTextureData);
	float value1 = (texture(aTextureData.data, coordinates1).x * range) + aMappedIntervalBands[0];
	float value2 = (texture(aTextureData.data, coordinates2).x * range) + aMappedIntervalBands[0];

	return applyIntegralTransferFunction1D( 
		vec2(value1, value2),
		aTextureData,
		aIntegralTransferFunction1D);
}

vec4 
transferFunction1D( 
		vec3 aPosition, 
		ImageData3D aTextureData,
		TransferFunction1D aTransferFunction1D,
		vec2 aMappedIntervalBands)
{
	vec3 coordinates = texCoordsFromPosition( aPosition, aTextureData );
	float range = aMappedIntervalBands[1] - aMappedIntervalBands[0];
	float value = (texture(aTextureData.data, coordinates).x * range) + aMappedIntervalBands[0];
		
	vec4 outputColor = applyTransferFunction1D( value, aTransferFunction1D );
	return outputColor;
}

vec4
doShading(
	vec3 aPosition,
	vec4 color,
	ImageData3D aTextureData,
	Light aLight,
	vec3 aEyePosition
	)
{
	vec3 coordinates = texCoordsFromPosition( aPosition, aTextureData );
	vec4 result = color;
	if (color.a > THRESHOLD) {
		vec3 gradient;
		COMPUTE_GRADIENT(gradient, value, aTextureData.data, coordinates);
		vec3 N = normalize( gradient );

		vec3 L = normalize( aLight.position - aPosition );
		vec3 V = normalize( aEyePosition - aPosition );

		Material material;
		material.Ka = vec3(0.1,0.1,0.1);
		material.Kd = vec3(0.6,0.6,0.6);
		material.Ks = vec3(0.2,0.2,0.2);
		material.shininess = 100;
		
		result.rgb += blinnPhongShading(N, V, L, material, aLight);
	}
	return result;
}

float 
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
	return clamp(
			(texture(aTextureData.data, coordinates).x - lowBand) * multiplier,
			0.0f,
			1.0f
			);
}
