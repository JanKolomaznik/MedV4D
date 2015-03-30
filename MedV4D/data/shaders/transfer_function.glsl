uniform Light gLight;
uniform TransferFunction1D gTransferFunction1D;
uniform TransferFunction2D gTransferFunction2D;

struct StepInfo {
	vec4 color;
	float previousValue;
};

vec4 
transferFunction1D( 
		vec3 aPosition, 
		ImageData3D aTextureData,
		TransferFunction1D aTransferFunction1D,
		vec2 aMappedIntervalBands)
{
	const bool EIGENVALUES_PREPROCESS_PRIMARY = true;
	
	vec3 coordinates = texCoordsFromPosition( aPosition, aTextureData );
	float range = aMappedIntervalBands[1] - aMappedIntervalBands[0];
	
	float value = 0;
	if (EIGENVALUES_PREPROCESS_PRIMARY)
	{
		//value = (computeVesselness(texture(aTextureData.data, coordinates).xyz) * range + aMappedIntervalBands[0]);
		float vesselness = computeVesselness(texture(aTextureData.data, coordinates).rgb);
		value = (vesselness * range) + aMappedIntervalBands[0];
	}
	else
	{
		value = (texture(aTextureData.data, coordinates).x * range) + aMappedIntervalBands[0];
	}
	
	return vec4(0, 1, 0, value);
	vec4 outputColor = applyTransferFunction1D( value, aTransferFunction1D );
	return outputColor;
}

vec4 
transferFunction2DWithSecondary( 
		vec3 aPosition, 
		ImageData3D aTextureData,
		vec2 aMappedIntervalBands,
		ImageData3D aSecondaryTextureData,
		vec2 aSecondaryMappedIntervalBands,
		TransferFunction2D aTransferFunction2D
		)
{
	vec3 coordinates = texCoordsFromPosition( aPosition, aTextureData );
	float range = aMappedIntervalBands[1] - aMappedIntervalBands[0];
	float value = 0;
	if (aTransferFunction2D.eigenvalueProcessPrimary)
	{
		vec3 eigenvalues = texture(aTextureData.data, coordinates).xyz;
		float vesselness = computeVesselness(eigenvalues, aTransferFunction2D.primaryEigenvalueParameters)*aTransferFunction2D.primaryValuesMultiplier;
		value = (vesselness * range) + aMappedIntervalBands[0];
	}
	else
	{
		value = (texture(aTextureData.data, coordinates).x * range) + aMappedIntervalBands[0];
	}
		
	vec3 coordinates2 = texCoordsFromPosition( aPosition, aSecondaryTextureData );
	float range2 = aSecondaryMappedIntervalBands[1] - aSecondaryMappedIntervalBands[0];
	float value2 = 0;
	if (aTransferFunction2D.eigenvalueProcessSecondary)
	{
		vec3 eigenvalues = texture(aSecondaryTextureData.data, coordinates2).xyz;
		float vesselness = computeVesselness(eigenvalues, aTransferFunction2D.secondaryEigenvalueParameters)*aTransferFunction2D.secondaryValuesMultiplier;
		value2 = ((vesselness * range2) + aSecondaryMappedIntervalBands[0]);
	}
	else
	{
		value2 = (texture(aSecondaryTextureData.data, coordinates2).x * range2) + aSecondaryMappedIntervalBands[0];
	}

	vec4 outputColor = applyTransferFunction2D(vec2(value, value2), aTransferFunction2D );
	return outputColor;
}


float 
getUnmappedValueVesselness(
		vec3 aPosition, 
		ImageData3D aTextureData,
		vec2 aMappedIntervalBands
		)
{
	float range = aMappedIntervalBands[1] - aMappedIntervalBands[0];
	vec3 coordinates = texCoordsFromPosition(aPosition, aTextureData);
	float vesselness = computeVesselness(texture(aTextureData.data, coordinates).xyz);
	
	//return 1;
	return (vesselness * range) + aMappedIntervalBands[0];
}

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
	if (sampleColor.a > 0.000001) {
		sampleColor.rgb *= 1.0 / sampleColor.a;
	}
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
#  ifdef USE_SECONDARY
	vec4 sampleColor = transferFunction2DWithSecondary(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands,
				gSecondaryImageData3D,
				gSecondaryMappedIntervalBands,
				gTransferFunction2D
				);
#  else	
	vec4 sampleColor = transferFunction2DWithGradient(
				aCoordinates,
				gPrimaryImageData3D,
				gMappedIntervalBands,
				gTransferFunction2D
				);
#  endif //USE_SECONDARY
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
	vec3 outColor = aInfo.color.rgb + sampleColor.rgb * sampleColor.a * (1 - alpha);
	alpha = alpha + sampleColor.a * (1 - alpha);
	aInfo.color = vec4(outColor, alpha);

	return aInfo;
}

vec4 colorFromStepInfo(StepInfo aInfo)
{
	return aInfo.color;
}

