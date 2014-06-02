#define BINARY_SEARCH_STEP_COUNT 4

uniform Light gLight;
uniform ImageData3D gPrimaryImageData3D;
uniform vec2 gMappedIntervalBands;
uniform float gIsoValue;
uniform vec4 gSurfaceColor;

struct StepInfo {
	vec4 color;
	bool isInside;
};

StepInfo initInfo(vec3 aCoordinates)
{
	StepInfo info;
	info.color = vec4(0.0, 0.0, 0.0, 0.0);
	info.isInside = gIsoValue < getUnmappedValue(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands
				);
	return info;
}

StepInfo doStep(StepInfo aInfo, vec3 aCoordinates, vec3 aRayDirection)
{
	float currentValue = getUnmappedValue(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands
				);
	bool isNowInside = gIsoValue < currentValue;
	if (isNowInside != aInfo.isInside) {
		float stepSize = -gRenderingSliceThickness / 2;
		float previousValue;

		for (int i = 0; i < BINARY_SEARCH_STEP_COUNT; ++i) {
			aCoordinates += aRayDirection * stepSize;
			previousValue = currentValue;
			currentValue = getUnmappedValue(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands
				);
			if (abs(previousValue - gIsoValue) < abs(currentValue - gIsoValue)) {
				stepSize /= -2;
			} else {
				stepSize /= 2;
			}
		}

		vec4 surfaceColor = vec4(0.5, 0.5, 0.2, 0.3);//gSurfaceColor;
		// SHADING ----------------------------------------------------------
		vec3 gradient = computeGradient(gPrimaryImageData3D, gMappedIntervalBands, currentValue, aCoordinates);
		vec3 N = normalize( gradient );

		vec3 L = normalize(gLight.position - aCoordinates);
		vec3 L2 = normalize((-1 * gLight.position) - aCoordinates);
		vec3 V = normalize(gCamera.eyePosition - aCoordinates);

		Material material;
		material.Ka = vec3(0.1,0.1,0.1);
		material.Kd = vec3(0.6,0.6,0.6);
		material.Ks = vec3(0.2,0.2,0.2);
		material.shininess = 100;
		
		surfaceColor.rgb += blinnPhongShading(N, V, L, material, gLight);
		surfaceColor.rgb += 0.8 * blinnPhongShading(N, V, L2, material, gLight);
		// -------------------------------------------------------------------
		float alpha = aInfo.color.a;
		vec3 outColor = aInfo.color.rgb/* * alpha*/ + surfaceColor.rgb * surfaceColor.a * (1 - alpha);
		alpha = alpha + surfaceColor.a * (1 - alpha);
		aInfo.color = vec4(outColor, alpha);
		aInfo.isInside = isNowInside;
	}

	return aInfo;
}

vec4 colorFromStepInfo(StepInfo aInfo)
{
	return aInfo.color;
}

