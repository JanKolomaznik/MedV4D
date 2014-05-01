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
	vec4 sampleColor = transferFunction1D(
			aCoordinates,
			gPrimaryImageData3D,
			gTransferFunction1D,
			gMappedIntervalBands);
	//sampleColor = vec4(0.0, 0.5, 0.0, 0.5);
	float alpha = aInfo.color.a;

	if (alpha < sampleColor.a) {
		aInfo.color = sampleColor;
	}

	//vec3 outColor = aInfo.color.rgb * alpha + sampleColor.rgb * sampleColor.a * (1 - alpha);
	//alpha = alpha + sampleColor.a * (1 - alpha);
	//aInfo.color = vec4(outColor, alpha);

	//aInfo.color = sampleColor;
	//aInfo.color = vec4(0.0, 0.5, 0.0, 0.5);
	return aInfo;
}

vec4 colorFromStepInfo(StepInfo aInfo)
{
	return vec4(aInfo.color.rgb, 0.8);
}

