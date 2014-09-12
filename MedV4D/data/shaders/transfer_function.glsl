uniform Light gLight;
uniform ImageData3D gPrimaryImageData3D;
uniform vec2 gMappedIntervalBands;
uniform TransferFunction1D gTransferFunction1D;
uniform TransferFunction2D gTransferFunction2D;

struct StepInfo {
	vec4 color;
	float previousValue;
};

StepInfo initInfo(vec3 aCoordinates)
{
	StepInfo info;
	info.color = vec4(0.0, 0.0, 0.0, 0.0);
	info.previousValue = getUnmappedValue(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands
				);
	return info;
}

StepInfo doStep(StepInfo aInfo, vec3 aCoordinates, vec3 aRayDirection)
{
#ifdef USE_TRANSFER_FUNCTION_1D
#  ifdef ENABLE_PREINTEGRATED_TRANSFER_FUNCTION
	float currentValue = getUnmappedValue(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands
				);

	vec4 sampleColor = applyIntegralTransferFunction1D( 
				vec2(aInfo.previousValue, currentValue),
				gPrimaryImageData3D,
				gTransferFunction1D
				);
	aInfo.previousValue = currentValue;
#  else	
	vec4 sampleColor = transferFunction1D(
				aCoordinates,
				gPrimaryImageData3D,
				gTransferFunction1D,
				gMappedIntervalBands);
#  endif //ENABLE_PREINTEGRATED_TRANSFER_FUNCTION
#endif //USE_TRANSFER_FUNCTION_1D


#ifdef USE_TRANSFER_FUNCTION_2D
	vec4 sampleColor = transferFunction2DWithGradient(
				aCoordinates,
				gPrimaryImageData3D,
				gTransferFunction2D,
				gMappedIntervalBands);
#endif //USE_TRANSFER_FUNCTION_2D


#ifdef ENABLE_SHADING
	sampleColor = doShading(
				aCoordinates,
				sampleColor,
				gPrimaryImageData3D,
				gLight,
				gCamera.eyePosition
				);
#endif //ENABLE_SHADING

	sampleColor.a = 1.0f - pow(1.0f - sampleColor.a, gRenderingSliceThickness);

	float alpha = aInfo.color.a;
	vec3 outColor = aInfo.color.rgb/* * alpha*/ + sampleColor.rgb * sampleColor.a * (1 - alpha);
	alpha = alpha + sampleColor.a * (1 - alpha);
	aInfo.color = vec4(outColor, alpha);

	return aInfo;
}

vec4 colorFromStepInfo(StepInfo aInfo)
{
	return aInfo.color;
}

