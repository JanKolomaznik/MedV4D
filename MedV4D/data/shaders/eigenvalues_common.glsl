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

float ExponencialFormula(float a, float b)
{
  return exp
    (-
      (
        (a*a) / (2 * b*b)
      )
    );
}

float computeVesselness(float lambda1, float lambda2, float lambda3, float alpha, float beta, float gamma)
{
	float returnValue = 0;
	
	switch (gEigenvaluesType)
	{
		case 0: // Franghi's
		{
			vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);
			if (sortedEigenvalues.y <= 0 && sortedEigenvalues.z <= 0)
			{
				float R_A = abs(sortedEigenvalues.y) / abs(sortedEigenvalues.z);
				float R_B = sqrt(abs(sortedEigenvalues.x) / abs(sortedEigenvalues.y*sortedEigenvalues.z));
				float S = sqrt(sortedEigenvalues.x*sortedEigenvalues.x + sortedEigenvalues.y*sortedEigenvalues.y + sortedEigenvalues.z*sortedEigenvalues.z);

				returnValue = (1 - ExponencialFormula(R_A, alpha)) * ExponencialFormula(R_B, beta) * (1 - ExponencialFormula(S, gamma));
			}
			else
			{
				returnValue = 0;
			}
		}
		break;
		
		case 1: // yoshinobu sato
		{
			vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);
			
			float lambdaC = min(-sortedEigenvalues.y, -sortedEigenvalues.z);
			
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
			
			returnValue = functionLambda1 * lambdaC / 10;
		}
		break;
		
		case 2:	// T.M. Koller, G. Gerig, G. Szekely
		{
			vec3 sortedEigenvalues = SortEigenValuesDecreasing(lambda1, lambda2, lambda3);
			if (sortedEigenvalues.y < 0 && sortedEigenvalues.z < 0 && sortedEigenvalues.y >= sortedEigenvalues.z)
			{
				returnValue = sqrt(abs(sortedEigenvalues.y * sortedEigenvalues.z));
			}
			else
			{
				returnValue = 0;
			}
		}
		break;
		
		case 3:
		{
			/*vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);
			
			const int imageDimension = 3;
			
			bool display = true;
			for (int i = gObjectDimension; i < imageDimension; ++i)
			{
				if (sortedEigenvalues[i] > 0)
				{
					display = false;
				}
			}
			
			if (display)
			{
				float R_A_denominator = 1;
				for (int i = gObjectDimension + 1; i < imageDimension; ++i)
				{
					//R_A_denominator *= abs(sortedEigenvalues[i]);
				}
				float R_A = abs(sortedEigenvalues[gObjectDimension]) / pow(R_A_denominator, 1/(imageDimension - gObjectDimension - 1));
				
				float R_B_denominator = 1;
				for (int i = gObjectDimension; i < imageDimension; ++i)
				{
					R_B_denominator *= abs(sortedEigenvalues[i]);
				}
				float R_B = abs(sortedEigenvalues[gObjectDimension-1]) / pow(R_B_denominator, 1/(imageDimension - gObjectDimension));
				
				float S = 0;
				for (int i = 0; i < imageDimension; ++i)
				{
					S += sortedEigenvalues[i]*sortedEigenvalues[i];
				}
				S = sqrt(S);
				
				returnValue = (1 - ExponencialFormula(R_A, alpha)) * ExponencialFormula(R_B, beta) * (1 - ExponencialFormula(S, gamma));
			}*/
		}
		break;
		
		case 4: // ours
		{
			vec3 sortedEigenvalues = SortEigenValuesAbsoluteValue(lambda1, lambda2, lambda3);
			
			if (sortedEigenvalues.y < 0 && sortedEigenvalues.z < 0 && abs(sortedEigenvalues.x) > alpha && abs(sortedEigenvalues.x) < beta)
			{
				returnValue = sortedEigenvalues.y/sortedEigenvalues.z;
			}
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