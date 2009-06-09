#ifndef MYFINITEDIFFERENCEFILTER_
#error File myFiniteDifferenceFilter.tcc cannot be included directly!
#else

#define DEBUGMYFINITEFILT 0

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage, class TOutputImage>
void
MyFiniteDifferenceImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
	// Test whether the output pixel type (or its components) are not of type
	// float or double:
	if( itk::NumericTraits< OutputPixelValueType>::is_integer )
	{
		itkWarningMacro("Output pixel type MUST be float or double to prevent computational errors");
	}

	if (this->GetState() == UNINITIALIZED)
	{
		// Allocate the output image
		this->AllocateOutputs();

		// Copy the input image to the output image.  Algorithms will operate
		// directly on the output image and the update buffer.
		this->CopyInputToOutput();

		// Perform any other necessary pre-iteration initialization.
		this->Initialize();

		this->SetStateToInitialized();
		m_ElapsedIterations = 0;
	}

	// Iterative algorithm
	TimeStepType dt;

	while ( ! this->Halt() )
	{
		//this->InitializeIteration(); // An optional method for precalculating
		// global values, or otherwise setting up
		// for the next iteration
		dt = this->CalculateChange();
		this->ApplyUpdate(dt);
		++m_ElapsedIterations;

		DL_PRINT(DEBUGMYFINITEFILT,
				"Elapsed iters:" << m_ElapsedIterations << ", dt = " << dt);

		if( (m_ElapsedIterations % 20) == 0)
		std::cout << "elapsed iters .. " << m_ElapsedIterations << std::endl;
	}

	if (m_ManualReinitialization == false)
	{
		this->SetStateToUninitialized(); // Reset the state once execution is
		// completed
	}
	// Any further processing of the solution can be done here.
	this->PostProcessOutput();
}

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage, class TOutputImage>
void
MyFiniteDifferenceImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
	// call the superclass' implementation of this method
	// copy the output requested region to the input requested region
	Superclass::GenerateInputRequestedRegion();

	//		// get pointers to the input
	//		typename Superclass::InputImagePointer inputPtr =
	//		const_cast< TInputImage *>( this->GetInput());
	//
	//		if ( !inputPtr )
	//		{
	//			return;
	//		}
	//
	//		// Get the size of the neighborhood on which we are going to operate.  This
	//		// radius is supplied by the difference function we are using.
	//		RadiusType radius = this->GetDifferenceFunction()->GetRadius();
	//
	//		// Try to set up a buffered region that will accommodate our
	//		// neighborhood operations.  This may not be possible and we
	//		// need to be careful not to request a region outside the largest
	//		// possible region, because the pipeline will give us whatever we
	//		// ask for.
	//
	//		// get a copy of the input requested region (should equal the output
	//		// requested region)
	//		typename TInputImage::RegionType inputRequestedRegion;
	//		inputRequestedRegion = inputPtr->GetRequestedRegion();
	//
	//		// pad the input requested region by the operator radius
	//		inputRequestedRegion.PadByRadius( radius );
	//
	//		//     std::cout << "inputRequestedRegion: " << inputRequestedRegion << std::endl;
	//		//     std::cout << "largestPossibleRegion: " << inputPtr->GetLargestPossibleRegion() << std::endl;
	//
	//		// crop the input requested region at the input's largest possible region
	//		if ( inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion()) )
	//		{
	//			inputPtr->SetRequestedRegion( inputRequestedRegion );
	//			return;
	//		}
	//		else
	//		{
	//			// Couldn't crop the region (requested region is outside the largest
	//			// possible region).  Throw an exception.
	//
	//			// store what we tried to request (prior to trying to crop)
	//			inputPtr->SetRequestedRegion( inputRequestedRegion );
	//
	//			// build an exception
	//			InvalidRequestedRegionError e(__FILE__, __LINE__);
	//			e.SetLocation(ITK_LOCATION);
	//			e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
	//			e.SetDataObject(inputPtr);
	//			throw e;
	//		}

}

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage, class TOutputImage>
TimeStepType
MyFiniteDifferenceImageFilter<TInputImage, TOutputImage>
::ResolveTimeStep(const TimeStepType *timeStepList, const bool *valid, int size)
{
	TimeStepType min;
	bool flag;
	min = itk::NumericTraits<TimeStepType>::Zero;

	// grab first valid value
	flag = false;
	for (int i = 0; i < size; ++i)
	{
		if (valid[i])
		{
			min = timeStepList[i];
			flag = true;
			break;
		}
	}

	if (!flag)
	{ // no values!
		throw itk::ExceptionObject(__FILE__, __LINE__);
	}

	// find minimum value
	for (int i = 0; i < size; ++i)
	{	if ( valid[i] && (timeStepList[i] < min) ) min = timeStepList[i];}

	return min;
}

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage, class TOutputImage>
bool
MyFiniteDifferenceImageFilter<TInputImage, TOutputImage>
::Halt()
{
	if (m_NumberOfIterations != 0)
	{
		this->UpdateProgress( static_cast<float>( this->GetElapsedIterations() ) /
				static_cast<float>( m_NumberOfIterations ) );
	}

	if (this->GetElapsedIterations() >= m_NumberOfIterations)
	{
		return true;
	}
	else if ( this->GetElapsedIterations() == 0)
	{
		return false;
	}
	else if ( this->GetMaximumRMSError()> m_RMSChange )
	{
		return true;
	}
	else
	{
		return false;
	}
}

///////////////////////////////////////////////////////////////////////////////
#endif
