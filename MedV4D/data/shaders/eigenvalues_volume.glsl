//uniform ImageData3D gPrimaryImageData3D;
//uniform vec2 gMappedIntervalBands;
uniform vec2 gWLWindow;

struct StepInfo {
	float value;
};

StepInfo initInfo(vec3 aCoordinates)
{
	StepInfo info;
	
	vec3 wlWindow = vec3(
		gWLWindow.x / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
		gWLWindow.y / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
		(gMappedIntervalBands[1] - gMappedIntervalBands[0]) / gWLWindow.x
		);
	vec3 encodedEigenvalues = applyWLWindowVector(
				aCoordinates,
				gPrimaryImageData3D,
				wlWindow
				);
				
  float lowBand = wlWindow.y - (wlWindow.x * 0.5f);
  float highBand = wlWindow.y + (wlWindow.x * 0.5f);
  float multiplier = wlWindow.z;
				
  vec3 eigenvalues = decodeEigenvalues(encodedEigenvalues);
  info.value = (computeVesselness(eigenvalues.x, eigenvalues.y, eigenvalues.z)*1000) * multiplier;
	
	return info;
}

StepInfo doStep(StepInfo aInfo, vec3 aCoordinates, vec3 aRayDirection)
{
	vec3 wlWindow = vec3(
		gWLWindow.x / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
		gWLWindow.y / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
		(gMappedIntervalBands[1] - gMappedIntervalBands[0]) / gWLWindow.x
		);
		
	vec3 encodedEigenvalues = applyWLWindowVector(
				aCoordinates,
				gPrimaryImageData3D,
				wlWindow
				);
				
  float lowBand = wlWindow.y - (wlWindow.x * 0.5f);
  float highBand = wlWindow.y + (wlWindow.x * 0.5f);
  float multiplier = wlWindow.z;
				
  vec3 eigenvalues = decodeEigenvalues(encodedEigenvalues);
  float value = (computeVesselness(eigenvalues.x, eigenvalues.y, eigenvalues.z)*1000) * multiplier;

#define ENABLE_MIP
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
	return vec4(1, 0, 0, aInfo.value);
}

