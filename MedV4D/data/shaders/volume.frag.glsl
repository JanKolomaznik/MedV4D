//#version 150
//precision highp float;

#define THRESHOLD (0.1)

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

struct Camera
{
	vec3 eyePosition;
	vec3 viewDirection;
	vec3 upDirection;	
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

uniform sampler2D gNoiseMap;
uniform float gJitterStrength = 1.0f;
uniform vec2 gNoiseMapSize;

uniform ImageData3D gPrimaryImageData3D;
//uniform ImageData3D gSecondaryImageData3D;

//int3 gImageDataResolution3D = {0, 0, 0};
uniform vec2 gMappedIntervalBands;

uniform Light gLight;
uniform Camera gCamera;

uniform TransferFunction1D gTransferFunction1D;
uniform float gRenderingSliceThickness;

/*vec3 blinnPhongShading(vec3 N, vec3 V, vec3 L, Material material, Light light)
{
	//half way vector
	vec3 H = normalize( L + V );

	//compute ambient term
	vec3 ambient = material.Ka * light.ambient;

	vec4 koef = lit(dot(N,L), dot(N, H), material.shininess);
	vec3 diffuse = material.Kd * light.color * koef.y;
	vec3 specular = material.Ks * light.color * koef.z;

	return ambient + diffuse + specular;
}*/

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
		vec3 position,
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

/*vec4
doShading(
	vec3 coordinates,
	vec4 color,
	ImageData3D aTextureData,
	Light aLight,
	vec3 aEyePosition
	)
{
	if (color.a > THRESHOLD) {
		vec3 gradient;
		COMPUTE_GRADIENT(gradient, value, aTextureData.data, coordinates); 
		vec3 N = normalize( gradient );

		vec3 L = normalize( aLight.position - position );
		vec3 V = normalize( aEyePosition - position );

		Material material;
		material.Ka = vec3(0.1,0.1,0.1);
		material.Kd = vec3(0.6,0.6,0.6);
		material.Ks = vec3(0.2,0.2,0.2);
		material.shininess = 100;
		
		OUT.color.rgb += BlinnPhongShading( N, V, L, material, aLight );
	}
	OUT.color = clamp( OUT.color, 0.0f.xxxx, 1.0f.xxxx);
	OUT.color.a = 1.0f - pow( 1.0f - OUT.color.a, aRenderingSliceThickness );
	return OUT;
}*/


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
		aPosition,
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
#endif

#ifdef ENABLE_PREINTEGRATED_TRANSFER_FUNCTION
	vec4 color = preintegratedTransferFunction1D(
				coordinates,
				gPrimaryImageData3D,
				gTransferFunction1D,
				gRenderingSliceThickness,
				gMappedIntervalBands,
				gCamera.viewDirection);
#else	
	vec4 color = transferFunction1D(
				coordinates,
				gPrimaryImageData3D,
				gTransferFunction1D,
				gMappedIntervalBands);
#endif
	color.a = 1.0f - pow(1.0f - color.a, gRenderingSliceThickness);
	fragmentColor = color;
}
