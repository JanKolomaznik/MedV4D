//uniform ImageData3D gPrimaryImageData3D;
//uniform vec2 gMappedIntervalBands;
uniform vec2 gWLWindow;

struct StepInfo {
	vec3 value;
};

StepInfo initInfo(vec3 aCoordinates)
{
	StepInfo info;
	info.value = applyWLWindowVector(
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
	vec3 value = applyWLWindowVector(
				aCoordinates,
				gPrimaryImageData3D,
				vec3( 
					gWLWindow.x / (gMappedIntervalBands[1] - gMappedIntervalBands[0]), 
					gWLWindow.y / (gMappedIntervalBands[1] - gMappedIntervalBands[0]),
					(gMappedIntervalBands[1] - gMappedIntervalBands[0]) / gWLWindow.x 
					)
				);

//#ifdef ENABLE_MIP
//	aInfo.value = max(aInfo.value, value);
//#else
//	value = 1.0f - pow(1.0f - value, gRenderingSliceThickness);
//	//value = 1.0 - exp(-gRenderingSliceThickness * (1.0 - value));
//	aInfo.value = aInfo.value + value * (1.0 - aInfo.value);
//#endif //ENABLE_MIP

  aInfo.value = value;

	return aInfo;
}

#define swap(a, b) float c = a; a = b; b = c;
#define clamp(value, min, max) (value < min) ? min : ((value > max) ? max : value);

vec3 SortEigenValuesAbsoluteValue(float lambda1, float lambda2, float lambda3)
{
  // simple bubble sort
  while (abs(lambda1) > abs(lambda2) || abs(lambda2) > abs(lambda3))
  {
    if (abs(lambda1) > abs(lambda2))
    {
      swap(lambda1, lambda2);
    }
    if (abs(lambda2) > abs(lambda3))
    {
      swap(lambda2, lambda3);
    }
  }

  return vec3(lambda1, lambda2, lambda3);
}

float ExponencialFormula(float a, float b)
{
  return exp
    (-
      (
        (a*a) / (2 * b*b)
      )
    );
}

float computeVesselness(float lambda1, float lambda2, float lambda3)
{
  const float alpha = 0.5;
  const float beta = 0.5;
  const float gamma = 2;

  vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);

  float R_A = abs(sortedEigenvalues.y) / abs(sortedEigenvalues.z);
  float R_B = abs(sortedEigenvalues.x) / abs(sortedEigenvalues.y*sortedEigenvalues.z);
  float S = sqrt(sortedEigenvalues.x*sortedEigenvalues.x + sortedEigenvalues.y*sortedEigenvalues.y + sortedEigenvalues.z*sortedEigenvalues.z);
	
  if (sortedEigenvalues.y < 0 && sortedEigenvalues.z < 0)
  {
    return (1 - ExponencialFormula(R_A, alpha)) * ExponencialFormula(R_B, beta) * (1 - ExponencialFormula(S, gamma));
  }
  else
  {
    return 0;
  }
}

float decodeEigenvalue(float encodedValue)
{
  const float NORMALIZATION_CONSTANT = 1000;

  return encodedValue <= NORMALIZATION_CONSTANT ? encodedValue : -(encodedValue - NORMALIZATION_CONSTANT) / NORMALIZATION_CONSTANT;
}

vec3 decodeEigenvalues(vec3 encodedValues)
{
  return vec3(decodeEigenvalue(encodedValues[0]), decodeEigenvalue(encodedValues[1]), decodeEigenvalue(encodedValues[2]));
}



vec4 colorFromStepInfo(StepInfo aInfo)
{
	// if (aInfo.value.x >= 1)
	// {
		// return vec4(1, 0, 0, 1);
	// }
	// else
	// {
		// return vec4(0, 1, 0, 1);
	// }
	// const float NORMALIZATION_VALUE = 1000;
	// return vec4(abs(aInfo.value.r / NORMALIZATION_VALUE), abs(aInfo.value.g / NORMALIZATION_VALUE), abs(aInfo.value.b / NORMALIZATION_VALUE), 0.5);

  vec3 eigenvalues = decodeEigenvalues(aInfo.value);
  float vesselness = computeVesselness(eigenvalues.x, eigenvalues.y, eigenvalues.z);
	return vec4(1, 1, 1, vesselness);
}

