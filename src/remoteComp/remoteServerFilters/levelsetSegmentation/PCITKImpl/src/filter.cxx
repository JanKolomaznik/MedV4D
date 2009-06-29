#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#error File filter.cxx cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::MySegmtLevelSetFilter(void)
{
}

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::PostProcessOutput()
{
	cntr_.Stop();
	
	std::cout << "Time measure stop  ..." << std::endl;
	Superclass::PostProcessOutput();
}

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::Initialize()
{
	Superclass::Initialize();
	std::cout << "Time measure start ..." << std::endl;
	
	cntr_.Reset();
	cntr_.Start();
}

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::PrintStats(std::ostream &s)
{
	s << "========= stats ===========" << std::endl;
	s << "Max. no. iterations: " << this->GetNumberOfIterations() << std::endl;
	s << "Max. RMS error: " << this->GetMaximumRMSError() << std::endl;
	s << "No. elpased iterations: " << this->GetElapsedIterations() << std::endl;
	s << "RMS change: " << this->GetRMSChange() << std::endl;
	s << std::endl;
	s << "Time measurement: " << cntr_ << std::endl;
	s << "===========================" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
#endif
