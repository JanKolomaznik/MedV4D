#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#error File updateCalculatorSPE.tcc cannot be included directly!
#else

namespace itk {

	///////////////////////////////////////////////////////////////////////////////

	template <class TInputImage,class TFeatureImage, class TOutputPixelType>
	void
	UpdateCalculatorSPE<TInputImage, TFeatureImage, TOutputPixelType>
	::Init(void)
	{
		MIN_NORM = 1.0e-6;

		double minSpacing = NumericTraits<double>::max();
		for (uint8 i=0; i<TInputImage::ImageDimension; i++)
		{
			minSpacing = vnl_math_min(minSpacing,m_Conf.m_inputImage->GetSpacing()[i]);
		}
		MIN_NORM *= minSpacing;
		
		// set props to diffFunc
		m_diffFunc->SetFeatureImage(m_Conf.m_featureImage);
		m_diffFunc->SetUpperThreshold(m_Conf.m_upThreshold);
		m_diffFunc->SetLowerThreshold(m_Conf.m_downThreshold);
		m_diffFunc->SetPropagationWeight(m_Conf.m_propWeight);
		m_diffFunc->SetCurvatureWeight(m_Conf.m_curvWeight);
	}

	///////////////////////////////////////////////////////////////////////////////
	template <class TInputImage,class TFeatureImage, class TOutputPixelType>
	void
	UpdateCalculatorSPE<TInputImage, TFeatureImage, TOutputPixelType>
	::CalculateChangeItem(NeighborhoodIterator<OutputImageType> &outIt)
	{
		FloatOffsetType offset;
		ValueType norm_grad_phi_squared, dx_forward, dx_backward, forwardValue,
		backwardValue, centerValue;
		uint32 i;

		// Calculate the offset to the surface from the center of this
		// neighborhood.  This is used by some level set functions in sampling a
		// speed, advection, or curvature term.
		if((centerValue = outIt.GetCenterPixel()) != 0.0 )
		{
			// Surface is at the zero crossing, so distance to surface is:
			// phi(x) / norm(grad(phi)), where phi(x) is the center of the
			// neighborhood.  The location is therefore
			// (i,j,k) - ( phi(x) * grad(phi(x)) ) / norm(grad(phi))^2
			norm_grad_phi_squared = 0.0;
			for (i = 0; i < TInputImage::ImageDimension; ++i)
			{
				forwardValue = outIt.GetNext(i);
				backwardValue = outIt.GetPrevious(i);

				if (forwardValue * backwardValue >= 0)
				{ //  Neighbors are same sign OR at least one neighbor is zero.
					dx_forward = forwardValue - centerValue;
					dx_backward = centerValue - backwardValue;

					// Pick the larger magnitude derivative.
					if (::vnl_math_abs(dx_forward)> ::vnl_math_abs(dx_backward) )
					{
						offset[i] = dx_forward;
					}
					else
					{
						offset[i] = dx_backward;
					}
				}
				else //Neighbors are opposite sign, pick the direction of the 0 surface.
				{
					if (forwardValue * centerValue < 0)
					{
						offset[i] = forwardValue - centerValue;
					}
					else
					{
						offset[i] = centerValue - backwardValue;
					}
				}

				norm_grad_phi_squared += offset[i] * offset[i];
			}

			for (i = 0; i < TInputImage::ImageDimension; ++i)
			{
				offset[i] = (offset[i] * centerValue) / (norm_grad_phi_squared + MIN_NORM);
			}

			m_Conf.m_UpdateBuffer->push_back( m_diffFunc->ComputeUpdate(outIt, (void *)&m_globalData, offset) );
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	template <class TInputImage,class TFeatureImage, class TOutputPixelType>
	typename UpdateCalculatorSPE<TInputImage, TFeatureImage, TOutputPixelType>::TimeStepType
	UpdateCalculatorSPE<TInputImage, TFeatureImage, TOutputPixelType>
	::CalculateChange()
	{
		typename SegmentationFunctionType::FloatOffsetType offset;
		  ValueType norm_grad_phi_squared, dx_forward, dx_backward, forwardValue,
		    backwardValue, centerValue;
		  unsigned i;
		  ValueType MIN_NORM      = 1.0e-6;
//		  if (this->GetUseImageSpacing())
//		    {
		    double minSpacing = NumericTraits<double>::max();
		    for (i=0; i< OutputImageType::ImageDimension; i++)
		      {
		      minSpacing = vnl_math_min(minSpacing, m_Conf.m_outputImage->GetSpacing()[i]);
		      }
		    MIN_NORM *= minSpacing;
//		    }

		  void *globalData = m_diffFunc->GetGlobalDataPointer();
		  
		  typename LayerType::ConstIterator layerIt;
		  NeighborhoodIterator<OutputImageType> outputIt(m_diffFunc->GetRadius(),
				  m_Conf.m_outputImage, m_Conf.m_outputImage->GetRequestedRegion());
		  TimeStepType timeStep;

		  const NeighborhoodScalesType neighborhoodScales = m_diffFunc->ComputeNeighborhoodScales();

//		  if ( m_BoundsCheckingActive == false )
//		    {
//		    outputIt.NeedToUseBoundaryConditionOff();
//		    }

		  // Calculates the update values for the active layer indicies in this
		  // iteration.  Iterates through the active layer index list, applying 
		  // the level set function to the output image (level set image) at each
		  // index.  Update values are stored in the update buffer.
		  for (layerIt = m_Conf.m_activeSet->Begin(); layerIt != m_Conf.m_activeSet->End(); ++layerIt)
		    {
		    outputIt.SetLocation(layerIt->m_Value);

		    // Calculate the offset to the surface from the center of this
		    // neighborhood.  This is used by some level set functions in sampling a
		    // speed, advection, or curvature term.
		    if ((centerValue = outputIt.GetCenterPixel()) != 0.0 )
		      {
		      // Surface is at the zero crossing, so distance to surface is:
		      // phi(x) / norm(grad(phi)), where phi(x) is the center of the
		      // neighborhood.  The location is therefore
		      // (i,j,k) - ( phi(x) * grad(phi(x)) ) / norm(grad(phi))^2
		      norm_grad_phi_squared = 0.0;
		      for (i = 0; i < OutputImageType::ImageDimension; ++i)
		        {
		        forwardValue  = outputIt.GetNext(i);
		        backwardValue = outputIt.GetPrevious(i);
		            
		        if (forwardValue * backwardValue >= 0)
		          { //  Neighbors are same sign OR at least one neighbor is zero.
		          dx_forward  = forwardValue - centerValue;
		          dx_backward = centerValue - backwardValue;

		          // Pick the larger magnitude derivative.
		          if (::vnl_math_abs(dx_forward) > ::vnl_math_abs(dx_backward) )
		            {
		            offset[i] = dx_forward;
		            }
		          else
		            {
		            offset[i] = dx_backward;
		            }
		          }
		        else //Neighbors are opposite sign, pick the direction of the 0 surface.
		          {
		          if (forwardValue * centerValue < 0)
		            {
		            offset[i] = forwardValue - centerValue;
		            }
		          else
		            {
		            offset[i] = centerValue - backwardValue;
		            }
		          }
		        
		        norm_grad_phi_squared += offset[i] * offset[i];
		        }
		      
		      for (i = 0; i < OutputImageType::ImageDimension; ++i)
		        {
		#if defined(ITK_USE_DEPRECATED_LEVELSET_INTERPOLATION)
		        offset[i] = (offset[i] * centerValue) * vcl_sqrt(ImageDimension +0.5) 
		                    / (norm_grad_phi_squared + MIN_NORM);
		#else
		        offset[i] = (offset[i] * centerValue) / (norm_grad_phi_squared + MIN_NORM);
		#endif
		        }
		          
		      m_Conf.m_UpdateBuffer->push_back( m_diffFunc->ComputeUpdate(outputIt, globalData, offset) );
		      }
		//    else // Don't do interpolation
		//      {
		//      m_UpdateBuffer.push_back( m_diffFunc->ComputeUpdate(outputIt, globalData) );
		//      }
		    }
		  
		  // Ask the finite difference function to compute the time step for
		  // this iteration.  We give it the global data pointer to use, then
		  // ask it to free the global data memory.
		  timeStep = m_diffFunc->ComputeGlobalTimeStep(globalData);

		  m_diffFunc->ReleaseGlobalDataPointer(globalData);
		  
		  return timeStep;
	}

	///////////////////////////////////////////////////////////////////////////////

} // namespace itk

#endif
