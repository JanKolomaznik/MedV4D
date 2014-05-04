uniform Light gLight;
uniform ImageData3D gPrimaryImageData3D;
uniform vec2 gMappedIntervalBands;
uniform TransferFunction1D gTransferFunction1D;

struct StepInfo {
	vec4 color;
};

StepInfo initInfo(vec3 aCoordinates)
{
	StepInfo info;
	info.color = transferFunction1D(
			aCoordinates,
			gPrimaryImageData3D,
			gTransferFunction1D,
			gMappedIntervalBands);
	return info;
}

StepInfo doStep(StepInfo aInfo, vec3 aCoordinates)
{
#ifdef ENABLE_PREINTEGRATED_TRANSFER_FUNCTION
	vec4 sampleColor = preintegratedTransferFunction1D(
				aCoordinates,
				gPrimaryImageData3D,
				gTransferFunction1D,
				gRenderingSliceThickness,
				gMappedIntervalBands,
				gCamera.viewDirection);
#else	
	vec4 sampleColor = transferFunction1D(
				aCoordinates,
				gPrimaryImageData3D,
				gTransferFunction1D,
				gMappedIntervalBands);
#endif //ENABLE_PREINTEGRATED_TRANSFER_FUNCTION

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

