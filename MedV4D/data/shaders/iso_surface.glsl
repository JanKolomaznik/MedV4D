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

StepInfo doStep(StepInfo aInfo, vec3 aCoordinates)
{
	float currentValue = getUnmappedValue(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands
				);
	bool isNowInside = gIsoValue < currentValue;

	if (isNowInside != aInfo.isInside) {
		float alpha = aInfo.color.a;
		vec3 outColor = aInfo.color.rgb/* * alpha*/ + gSurfaceColor.rgb * gSurfaceColor.a * (1 - alpha);
		alpha = alpha + gSurfaceColor.a * (1 - alpha);
		aInfo.color = vec4(outColor, alpha);
	}
#ifdef ENABLE_SHADING
	sampleColor = doShading(
				aCoordinates,
				sampleColor,
				gPrimaryImageData3D,
				gLight,
				gCamera.eyePosition
				);
#endif //ENABLE_SHADING

	return aInfo;
}

vec4 colorFromStepInfo(StepInfo aInfo)
{
	return aInfo.color;
}

