#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#error File updateCalculatorSPE.tcc cannot be included directly!
#else

// support functions
template<typename ImageType, typename RegionType>
RegionType ConvertRegion(const ImageType &image)
{
	// convert values
	typename ImageType::RegionType imageRegion;
	typename RegionType::OffsetType offset;
	typename RegionType::SizeType size;

	imageRegion = image.GetLargestPossibleRegion();

	for(uint8 i=0; i<ImageType::ImageDimension; i++)
	{
		offset[i] = imageRegion.GetIndex()[i];
		size[i] = imageRegion.GetSize()[i];
	}

	RegionType reg(offset, size);

	return reg;
}

template<typename ImageType>
void 
PrintITKImage(const ImageType &image, std::ostream &s)
{
	image.Print( s);
	    
	typename ImageType::RegionType::IndexType index;
	typename ImageType::RegionType::SizeType size = 
    	image.GetLargestPossibleRegion().GetSize();
    
    s << "size: " << size[0] << "," << size[1] << "," << size[2] << std::endl;
    
    for( unsigned int i=0; i<size[0]; i++)
    {
    	for( unsigned int j=0; j<size[1]; j++)
    	{
    		for( unsigned int k=0; k< size[2]; k++)
    		{
    			index[0] = i;
    			index[1] = j;
    			index[2] = k;
    			
    			s << "[" << i << "," << j << "," << k << "]= ";
    			s << image.GetPixel(index) << std::endl;
    		}
    	}
    }
}

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
void
UpdateCalculatorSPE<TInputImage, TFeatureImage, TOutputPixelType>
::Init(void)
{
	MIN_NORM = 1.0e-6;

	double minSpacing = itk::NumericTraits<double>::max();
	for (uint8 i=0; i<TInputImage::ImageDimension; i++)
	{
		minSpacing = vnl_math_min(minSpacing,m_Conf.m_inputImage->GetSpacing()[i]);
	}
	MIN_NORM *= minSpacing;

	// set props to diffFunc
	m_diffFunc.SetUpperThreshold(m_Conf.m_upThreshold);
	m_diffFunc.SetLowerThreshold(m_Conf.m_downThreshold);
	m_diffFunc.SetPropagationWeight(m_Conf.m_propWeight);
	m_diffFunc.SetCurvatureWeight(m_Conf.m_curvWeight);
}

///////////////////////////////////////////////////////////////////////////////
//template <class TInputImage,class TFeatureImage, class TOutputPixelType>
//void
//UpdateCalculatorSPE<TInputImage, TFeatureImage, TOutputPixelType>
//::CalculateChangeItem(NeighborhoodIterator<OutputImageType> &outIt)
//{
//	FloatOffsetType offset;
//	ValueType norm_grad_phi_squared, dx_forward, dx_backward, forwardValue,
//	backwardValue, centerValue;
//	uint32 i;
//
//	// Calculate the offset to the surface from the center of this
//	// neighborhood.  This is used by some level set functions in sampling a
//	// speed, advection, or curvature term.
//	if((centerValue = outIt.GetCenterPixel()) != 0.0 )
//	{
//		// Surface is at the zero crossing, so distance to surface is:
//		// phi(x) / norm(grad(phi)), where phi(x) is the center of the
//		// neighborhood.  The location is therefore
//		// (i,j,k) - ( phi(x) * grad(phi(x)) ) / norm(grad(phi))^2
//		norm_grad_phi_squared = 0.0;
//		for (i = 0; i < TInputImage::ImageDimension; ++i)
//		{
//			forwardValue = outIt.GetNext(i);
//			backwardValue = outIt.GetPrevious(i);
//
//			if (forwardValue * backwardValue >= 0)
//			{ //  Neighbors are same sign OR at least one neighbor is zero.
//				dx_forward = forwardValue - centerValue;
//				dx_backward = centerValue - backwardValue;
//
//				// Pick the larger magnitude derivative.
//				if (::vnl_math_abs(dx_forward)> ::vnl_math_abs(dx_backward) )
//				{
//					offset[i] = dx_forward;
//				}
//				else
//				{
//					offset[i] = dx_backward;
//				}
//			}
//			else //Neighbors are opposite sign, pick the direction of the 0 surface.
//			{
//				if (forwardValue * centerValue < 0)
//				{
//					offset[i] = forwardValue - centerValue;
//				}
//				else
//				{
//					offset[i] = centerValue - backwardValue;
//				}
//			}
//
//			norm_grad_phi_squared += offset[i] * offset[i];
//		}
//
//		for (i = 0; i < TInputImage::ImageDimension; ++i)
//		{
//			offset[i] = (offset[i] * centerValue) / (norm_grad_phi_squared + MIN_NORM);
//		}
//
//		m_Conf.m_UpdateBuffer->push_back( m_diffFunc.ComputeUpdate(outIt, (void *)&m_globalData, offset) );
//	}
//}

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
	ValueType MIN_NORM = 1.0e-6;
	//		  if (this->GetUseImageSpacing())
	//		    {
	double minSpacing = itk::NumericTraits<double>::max();
	for (i=0; i< OutputImageType::ImageDimension; i++)
	{
		minSpacing = vnl_math_min(minSpacing, m_Conf.m_outputImage->GetSpacing()[i]);
	}
	MIN_NORM *= minSpacing;
	//		    }

	typename LayerType::ConstIterator layerIt;

	typedef NeighborhoodCell<typename OutputImageType::PixelType, OutputImageType::ImageDimension> OutNeighbourhood;
	typedef NeighborhoodCell<FeaturePixelType, OutputImageType::ImageDimension> FeatureNeighbourhood;

	// fill the image properties
	typename OutNeighbourhood::TImageProperties outImageProps(
			ConvertRegion<OutputImageType, typename OutNeighbourhood::TRegion>(*m_Conf.m_outputImage),
			(InputPixelType *)m_Conf.m_outputImage->GetBufferPointer() );
	typename FeatureNeighbourhood::TImageProperties featureImageProps(
			ConvertRegion<TFeatureImage, typename FeatureNeighbourhood::TRegion>(*m_Conf.m_featureImage),
			(FeaturePixelType *)m_Conf.m_featureImage->GetBufferPointer() );

	// create neghbours as middle layer between image in PPE and part of image on SPE
	OutNeighbourhood outNeigh(m_diffFunc.GetRadius(), &outImageProps);
	FeatureNeighbourhood featureNeigh(m_diffFunc.GetRadius(), &featureImageProps);

	// create neigbor iterators to perform calculations on
	typedef NeighbourIteratorCell<typename OutputImageType::PixelType, OutputImageType::ImageDimension> TOutIter;
	typedef NeighbourIteratorCell<FeaturePixelType, OutputImageType::ImageDimension> TFeatureIter;
	TOutIter outIter(&outNeigh);
	TFeatureIter featureIter(&featureNeigh);

	TimeStepType timeStep;

	const NeighborhoodScalesType neighborhoodScales = m_diffFunc.ComputeNeighborhoodScales();

	//		  if ( m_BoundsCheckingActive == false )
	//		    {
	//		    outputIt.NeedToUseBoundaryConditionOff();
	//		    }

	//PrintITKImage<OutputImageType>(*m_Conf.m_outputImage,LOUT);
	
	// Calculates the update values for the active layer indicies in this
	// iteration.  Iterates through the active layer index list, applying 
	// the level set function to the output image (level set image) at each
	// index.  Update values are stored in the update buffer.
	for (layerIt = m_Conf.m_activeSet->Begin(); layerIt != m_Conf.m_activeSet->End(); ++layerIt)
	{
		outIter.SetLocation((typename TOutIter::IndexType &)layerIt->m_Value);
		featureIter.SetLocation((typename TFeatureIter::IndexType &)layerIt->m_Value);
		
		//outIter.GetNeighborhood().Print(LOUT);
		

		// Calculate the offset to the surface from the center of this
		// neighborhood.  This is used by some level set functions in sampling a
		// speed, advection, or curvature term.
		if ((centerValue = outIter.GetCenterPixel()) != 0.0 )
		{
			// Surface is at the zero crossing, so distance to surface is:
			// phi(x) / norm(grad(phi)), where phi(x) is the center of the
			// neighborhood.  The location is therefore
			// (i,j,k) - ( phi(x) * grad(phi(x)) ) / norm(grad(phi))^2
			norm_grad_phi_squared = 0.0;
			for (i = 0; i < OutputImageType::ImageDimension; ++i)
			{
				forwardValue = outIter.GetNext(i);
				backwardValue = outIter.GetPrevious(i);

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

			for (i = 0; i < OutputImageType::ImageDimension; ++i)
			{
#if defined(ITK_USE_DEPRECATED_LEVELSET_INTERPOLATION)
				offset[i] = (offset[i] * centerValue) * vcl_sqrt(ImageDimension +0.5)
				/ (norm_grad_phi_squared + MIN_NORM);
#else
				offset[i] = (offset[i] * centerValue) / (norm_grad_phi_squared + MIN_NORM);
#endif
			}

			m_Conf.m_UpdateBuffer->push_back( m_diffFunc.ComputeUpdate(outIter, featureIter, (void*)&m_globalData, offset) );
		}
		//    else // Don't do interpolation
		//      {
		//      m_UpdateBuffer.push_back( m_diffFunc.ComputeUpdate(outIter, globalData) );
		//      }
	}

	// Ask the finite difference function to compute the time step for
	// this iteration.  We give it the global data pointer to use, then
	// ask it to free the global data memory.
	timeStep = m_diffFunc.ComputeGlobalTimeStep(&m_globalData);

	return timeStep;
}

///////////////////////////////////////////////////////////////////////////////

#endif
