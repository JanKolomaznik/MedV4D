#ifndef MYFINITEDIFFERENCEFILTER_
#error File myFiniteDifferenceFilter.tcc cannot be included directly!
#else

#define DEBUGMYFINITEFILT 12

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

#ifndef NO_COPY_INPUT_TO_OUTPUT
		// Copy the input image to the output image.  Algorithms will operate
		// directly on the output image and the update buffer.
		this->CopyInputToOutput();
#endif

		// Perform any other necessary pre-iteration initialization.
		this->Initialize();

		this->SetStateToInitialized();
		m_ElapsedIterations = 0;
	}

	// Iterative algorithm
	TimeStepType dt;

	while ( ! this->Halt() )
	{
		dt = this->CalculateChange();
		this->ApplyUpdate(dt);
		++m_ElapsedIterations;

		LOG("Elapsed iters:" << m_ElapsedIterations << ", dt = " << dt);
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
