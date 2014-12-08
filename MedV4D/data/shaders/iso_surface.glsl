#define BINARY_SEARCH_STEP_COUNT 4

uniform Light gLight;

struct IsoSurfaceDefinition {
	float isoValue;
	vec4 surfaceColor;
};

//uniform float gIsoValue;
//uniform vec4 gSurfaceColor;

//uniform IsoSurfaceDefinition gIsoSurfaces[ISO_SURFACES_COUNT];
uniform float gIsoValues[ISO_SURFACES_COUNT];
uniform vec4 gIsoSurfacesColors[ISO_SURFACES_COUNT];

struct StepInfo {
	vec4 color;
	//bool isInside;
	bool isInside[ISO_SURFACES_COUNT];
};

StepInfo
processIsoSurface(StepInfo aInfo, vec4 aSurfaceColor, float aIsoValue, float aCurrentValue, vec3 aCoordinates, vec3 aRayDirection)
{
	float stepSize = -gRenderingSliceThickness / 2;
	float previousValue;
	float currentValue = aCurrentValue;

	for (int i = 0; i < BINARY_SEARCH_STEP_COUNT; ++i) {
		aCoordinates += aRayDirection * stepSize;
		previousValue = currentValue;
		currentValue = getUnmappedValue(
			aCoordinates,
			gPrimaryImageData3D,
			gMappedIntervalBands
			);
		if (abs(previousValue - aIsoValue) < abs(currentValue - aIsoValue)) {
			stepSize /= -2;
		} else {
			stepSize /= 2;
		}
	}

	vec4 surfaceColor = aSurfaceColor;
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
	vec3 outColor = aInfo.color.rgb + surfaceColor.rgb * surfaceColor.a * (1 - alpha);
	alpha = alpha + surfaceColor.a * (1 - alpha);
	aInfo.color = vec4(outColor, alpha);

	return aInfo;
}

StepInfo initInfo(vec3 aCoordinates)
{
	//float gIsoValue = gIsoValues[1];
	//vec4 gSurfaceColor = gIsoSurfacesColors[1];
	StepInfo info;
	info.color = vec4(0.0, 0.0, 0.0, 0.0);
	float value = getUnmappedValue(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands
				);
	//info.isInside = gIsoValue < value;

	for (int i = 0; i < ISO_SURFACES_COUNT; ++i) {
		info.isInside[i] = gIsoValues[i] < value;
	}
	return info;
}

StepInfo doStep(StepInfo aInfo, vec3 aCoordinates, vec3 aRayDirection)
{
	//float gIsoValue = gIsoValues[1];
	//vec4 gSurfaceColor = gIsoSurfacesColors[1];
	float currentValue = getUnmappedValue(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands
				);
	for (int i = 0; i < ISO_SURFACES_COUNT; ++i) {
		bool isNowInside = gIsoValues[i] < currentValue;
		
		if (isNowInside != aInfo.isInside[i]) {
			aInfo = processIsoSurface(
				aInfo, 
				gIsoSurfacesColors[i], 
				gIsoValues[i], 
				currentValue, 
				aCoordinates, 
				aRayDirection);
			aInfo.isInside[i] = isNowInside;
		}
	}
	return aInfo;
	/*bool isNowInside = gIsoValue < currentValue;
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

		vec4 surfaceColor = gSurfaceColor;//gSurfaceColor;
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
		vec3 outColor = aInfo.color.rgb + surfaceColor.rgb * surfaceColor.a * (1 - alpha);
		alpha = alpha + surfaceColor.a * (1 - alpha);
		aInfo.color = vec4(outColor, alpha);
		aInfo.isInside = isNowInside;
	}*/

	return aInfo;
}

vec4 colorFromStepInfo(StepInfo aInfo)
{
	return aInfo.color;
}

