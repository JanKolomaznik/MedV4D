uniform vec3 gEigenvaluesConstants;
uniform int gEigenvaluesType;
uniform int gObjectDimension;

#define swap(a, b) float c = a; a = b; b = c;
//#define clamp(value, min, max) (value < min) ? min : ((value > max) ? max : value);

vec3 SortEigenValuesDecreasing(float lambda1, float lambda2, float lambda3)
{
  // simple bubble sort
  while (lambda1 < lambda2 || lambda2 < lambda3)
  {
    if (lambda1 < lambda2)
    {
      swap(lambda1, lambda2);
    }
    if (lambda2 < lambda3)
    {
      swap(lambda2, lambda3);
    }
  }

  return vec3(lambda1, lambda2, lambda3);
}

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

float accessVectorElement(vec3 vector, int index)
{
	// this is just horrible workaround that has to be done in order to got this working
	// for some reason, element access by variable is not working (tested on two different graphic cards)
	
	if (index == 0)
	{
		return vector.x;
	}
	else if (index == 1)
	{
		return vector.y;
	}
	else if (index == 2)
	{
		return vector.z;
	}
	else
	{
		return 0;
	}
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

float computeFranghiVesselness(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma)
{
	vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);
	if (sortedEigenvalues.y <= 0 && sortedEigenvalues.z <= 0)
	{
		float R_A = abs(sortedEigenvalues.y) / abs(sortedEigenvalues.z);
		float R_B = abs(sortedEigenvalues.x) / sqrt(abs(sortedEigenvalues.y*sortedEigenvalues.z));
		float S = sqrt(sortedEigenvalues.x*sortedEigenvalues.x + sortedEigenvalues.y*sortedEigenvalues.y + sortedEigenvalues.z*sortedEigenvalues.z);

		return (1 - ExponencialFormula(R_A, alpha)) * ExponencialFormula(R_B, beta) * (1 - ExponencialFormula(S, gamma));
	}
	else
	{
		return 0;
	}
}

float computeSatoVesselness(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma)
{
	vec3 sortedEigenvalues = SortEigenValuesDecreasing(lambda1, lambda2, lambda3);
	
	float lambdaC = -sortedEigenvalues.y;
	
	float functionLambda1 = 0;
	
	if (lambdaC > 0)
	{
		if (sortedEigenvalues.x <= 0)
		{
			functionLambda1 = exp(-((sortedEigenvalues.x*sortedEigenvalues.x)/(2*alpha*alpha*lambdaC*lambdaC)));
		}
		else
		{
			functionLambda1 = exp(-((sortedEigenvalues.x*sortedEigenvalues.x)/(2*beta*beta*lambdaC*lambdaC)));
		}
	}
	else
	{
		functionLambda1 = 0;
	}
	
	return functionLambda1 * lambdaC / 10;
}

float computeKollerVesselness(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma)
{
	vec3 sortedEigenvalues = SortEigenValuesDecreasing(lambda1, lambda2, lambda3);
	
	if (sortedEigenvalues.y < 0 && sortedEigenvalues.z < 0 && sortedEigenvalues.y >= sortedEigenvalues.z)
	{
		return sqrt(abs(sortedEigenvalues.y * sortedEigenvalues.z));
	}
	else
	{
		return 0;
	}
}

float computeCustomVesselness(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma)
{
	vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);
			
	if (sortedEigenvalues.y < 0 && sortedEigenvalues.z < 0 && abs(sortedEigenvalues.x) > alpha && abs(sortedEigenvalues.x) < beta)
	{
		float norm = sqrt(sortedEigenvalues.x*sortedEigenvalues.x + sortedEigenvalues.y*sortedEigenvalues.y + sortedEigenvalues.z*sortedEigenvalues.z);
		
		return (sortedEigenvalues.y / sortedEigenvalues.z) * norm;
	}
	else
	{
		return 0;
	}
}

float computeBlobness(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma)
{
	vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);

	
	if (sortedEigenvalues.x < 0 && sortedEigenvalues.y < 0 && sortedEigenvalues.z < 0)
	{
		float R_A_denominator = abs(sortedEigenvalues.y*sortedEigenvalues.z);
		float R_A = abs(sortedEigenvalues.x) / sqrt(R_A_denominator);
		
		float R_B = 0;
		
		float S = sqrt(lambda1*lambda1+lambda2*lambda2+lambda3*lambda3);
		
		return (1 - ExponencialFormula(R_A, alpha)) * ExponencialFormula(R_B, beta) * (1 - ExponencialFormula(S, gamma));
	}
	else
	{
		return 0;
	}
	
	return 0;
}

float computePlateness(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma)
{
	vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);
	
	if (sortedEigenvalues.z < 0)
	{
		float R_A = 1.0 / 0.0; // infinity
		
		float R_B = abs(sortedEigenvalues.y) / abs(sortedEigenvalues.z);
		
		float S = sqrt(lambda1*lambda1+lambda2*lambda2+lambda3*lambda3);
		
		return (1 - ExponencialFormula(R_A, alpha)) * ExponencialFormula(R_B, beta) * (1 - ExponencialFormula(S, gamma));
	}
	else
	{
		return 0;
	}
}

float computeObjectnessDirectly(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma, int objectDimension)
{
	vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);
			
	const int imageDimension = 3;
	
	bool display = true;
	for (int i = objectDimension; i < imageDimension; ++i)
	{
		if (accessVectorElement(sortedEigenvalues, i) >= 0)
		{
			display = false;
		}
	}
	
	if (display)
	{
		float R_A_denominator = 1;
		for (int i = objectDimension + 1; i < imageDimension; ++i)
		{
			R_A_denominator *= abs(accessVectorElement(sortedEigenvalues, i));
		}
		float R_A = abs(accessVectorElement(sortedEigenvalues, objectDimension)) / pow(R_A_denominator, 1/(imageDimension - objectDimension - 1));
		
		float R_B_denominator = 1;
		for (int i = objectDimension; i < imageDimension; ++i)
		{
			R_B_denominator *= abs(accessVectorElement(sortedEigenvalues, i));
		}
		float R_B = abs(accessVectorElement(sortedEigenvalues, objectDimension-1)) / pow(R_B_denominator, 1/(imageDimension - objectDimension));
		
		float S = 0;
		for (int i = 0; i < imageDimension; ++i)
		{
			float square = accessVectorElement(sortedEigenvalues, i);
			S += square*square;
		}
		S = sqrt(S);
		
		return (1 - ExponencialFormula(R_A, alpha)) * ExponencialFormula(R_B, beta) * (1 - ExponencialFormula(S, gamma));
	}
	else
	{
		return 0;
	}
}

float computeObjectness(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma, int objectDimension)
{
	// unfortunately, direct computation suffers from numerical imprecisions
	//return computeObjectnessDirectly(lambda1, lambda2, lambda3, alpha, beta, gamma, objectDimension);
	
	
	if (objectDimension == 0)
	{
		return computeBlobness(lambda1, lambda2, lambda3, alpha, beta, gamma);
	}
	else if (objectDimension == 1)
	{
		return computeFranghiVesselness(lambda1, lambda2, lambda3, alpha, beta, gamma);
	}
	else if (objectDimension == 2)
	{
		return computePlateness(lambda1, lambda2, lambda3, alpha, beta, gamma);
	}
	else
	{
		return 0;
	}
}

float computeVesselness(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma)
{
	float returnValue = 0;
	
	switch (gEigenvaluesType)
	{
		case 0: // Franghi's
		{
			returnValue = computeFranghiVesselness(lambda1, lambda2, lambda3, alpha, beta, gamma);
		}
		break;
		
		case 1: // yoshinobu sato
		{
			returnValue = computeSatoVesselness(lambda1, lambda2, lambda3, alpha, beta, gamma);
		}
		break;
		
		case 2:	// T.M. Koller, G. Gerig, G. Szekely
		{
			returnValue = computeKollerVesselness(lambda1, lambda2, lambda3, alpha, beta, gamma);
		}
		break;
		
		case 3: // objectness
		{			
			returnValue = computeObjectness(lambda1, lambda2, lambda3, alpha, beta, gamma, gObjectDimension);
		}
		break;
		
		case 4: // ours
		{
			returnValue = computeCustomVesselness(lambda1, lambda2, lambda3, alpha, beta, gamma);
		}
		break;
	}
	
	return returnValue;
}

float computeVesselness(float lambda1, float lambda2, float lambda3)
{
	return computeVesselness(lambda1, lambda2, lambda3, gEigenvaluesConstants.x, gEigenvaluesConstants.y, gEigenvaluesConstants.z);
}

float computeVesselness(vec3 eigenvalues)
{
	return computeVesselness(eigenvalues[0], eigenvalues[1], eigenvalues[2]);
}

float computeVesselness(vec3 eigenvalues, vec3 parameters)
{
	return computeVesselness(eigenvalues[0], eigenvalues[1], eigenvalues[2], parameters[0], parameters[1], parameters[2]);
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