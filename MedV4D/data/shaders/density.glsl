//uniform ImageData3D gPrimaryImageData3D;
//uniform vec2 gMappedIntervalBands;
uniform vec2 gWLWindow;

struct StepInfo {
	float value;
};

StepInfo initInfo(vec3 aCoordinates)
{
	StepInfo info;
	info.value = applyWLWindow(
				aCoordinates,
				gPrimaryImageData3D,
				vec3(
					gWLWindow.x / (gMappedIntervalBands[1] - gMappedIntervalBands[0]), 
					gWLWindow.y / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
					(gMappedIntervalBands[1] - gMappedIntervalBands[0]) / gWLWindow.x 
					)
				); 
	return info;
}

StepInfo doStep(StepInfo aInfo, vec3 aCoordinates, vec3 aRayDirection)
{
	float value = applyWLWindow(
				aCoordinates,
				gPrimaryImageData3D,
				vec3( 
					gWLWindow.x / (gMappedIntervalBands[1] - gMappedIntervalBands[0]), 
					gWLWindow.y / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
					(gMappedIntervalBands[1] - gMappedIntervalBands[0]) / gWLWindow.x 
					)
				);

#ifdef ENABLE_MIP
	aInfo.value = max(aInfo.value, value);
#else
	value = 1.0f - pow(1.0f - value, gRenderingSliceThickness);
	//value = 1.0 - exp(-gRenderingSliceThickness * (1.0 - value));
	aInfo.value = aInfo.value + value * (1.0 - aInfo.value);
#endif //ENABLE_MIP

	return aInfo;
}

vec4 colorFromStepInfo(StepInfo aInfo)
{
	return vec4(1.0, 1.0, 1.0, aInfo.value);
}

