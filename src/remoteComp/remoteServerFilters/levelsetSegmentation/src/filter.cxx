#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#error File filter.cxx cannot be included directly!
#else

namespace itk {

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
ThreshSegLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::ThreshSegLevelSetFilter(void)
{
	func_ = SegmentationFunctionType::New();
	this->SetDifferenceFunction(func_);
	
	  this->SetIsoSurfaceValue(NumericTraits<ValueType>::Zero);
	  
	  // Provide some reasonable defaults which will at least prevent infinite
	  // looping.
	  this->SetMaximumRMSError(0.02);
	  this->SetNumberOfIterations(1000);
}

///////////////////////////////////////////////////////////////////////////////

//template <class TInputImage,class TFeatureImage, class TOutputPixelType>
//void
//ThreshSegLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
//::Initialize(void)
//{
//	Superclass::Initialize();
//	
//	func_->cntr_.Reset();
//	cntr_.Reset();
//}

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
void
ThreshSegLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::GenerateData(void)
{
	//func_->cntr_.Reset();
	cntr_.Reset();
	cntr_.Start();

	  // A positive speed value causes surface expansion, the opposite of the
	  // default.  Flip the sign of the propagation and advection weights.
//	  if (m_ReverseExpansionDirection == true)
//	    {
//	    this->GetSegmentationFunction()->ReverseExpansionDirection();
//	    }
	  
	  // Start the solver
	  Superclass::GenerateData();
	  
	  // Reset all the signs of the weights.
//	  if (m_ReverseExpansionDirection == true)
//	    {
//	    this->GetSegmentationFunction()->ReverseExpansionDirection();
//	    }  
	  
	cntr_.Stop();
}

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
void
ThreshSegLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::PrintStats(std::ostream &s)
{
	s << "========= stats ===========" << std::endl;
	s << "Max. no. iterations: " << this->GetNumberOfIterations() << std::endl;
	s << "Max. RMS error: " << this->GetMaximumRMSError() << std::endl;
	s << "No. elpased iterations: " << this->GetElapsedIterations() << std::endl;
	s << "RMS change: " << this->GetRMSChange() << std::endl;
	s << std::endl;
	s << "Time spent in solver: " << cntr_ << std::endl;
	//s << "Time spent in difference solving: " << func_->cntr_ << std::endl;
	s << "===========================" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
template <class TInputImage,class TFeatureImage, class TOutputPixelType>
typename ThreshSegLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>::TimeStepType
ThreshSegLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
	::CalculateChange(void)
{
	//std::cout << "Sem tu!" << std::endl;
	
	return Superclass::CalculateChange();
}

///////////////////////////////////////////////////////////////////////////////
template <class TInputImage,class TFeatureImage, class TOutputPixelType>
void
ThreshSegLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::ApplyUpdate(TimeStepType dt)
{
	//std::cout << "Sem tu!" << std::endl;
	Superclass::ApplyUpdate(dt);
}

///////////////////////////////////////////////////////////////////////////////
}
#endif
