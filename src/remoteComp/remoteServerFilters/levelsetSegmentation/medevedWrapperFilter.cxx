
#ifndef SERVERLEVELSETSEGMENTATION_H_
#error File medevedWrapperFilter.cxx cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::ThreshLSSegMedvedWrapper(Properties *props)
	: properties_(props)
	, _levelSetImageData(NULL)
{
	this->_name = "KlecLevelSet Filter";
		
		// crete seed node as begining for fast marching filter		  
		FastMarchingFilterType::NodeType node;
		  
		  _seeds = NodeContainer::New();
		  _seeds->Initialize();
		  _seeds->InsertElement( 0, node );
		  
		  _levelSetImage = TLevelSetImage::New();
		  
	// setup filters
	SetupBinaryThresholder();
	  
	// connect the pipeline into in/out of the ITKFilter
	SetOutputITKImage( thresholder->GetOutput() );
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
::~ThreshLSSegMedvedWrapper()
{
	if(_levelSetImageData) free(_levelSetImageData);
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::RunFastMarchingFilter(void)
{
	FastMarchingFilterType::Pointer fastMarching = FastMarchingFilterType::New();
  //
  //  The FastMarchingImageFilter requires the user to provide a seed
  //  point from which the level set will be generated. The user can actually
  //  pass not only one seed point but a set of them. Note the the
  //  FastMarchingImageFilter is used here only as a helper in the
  //  determination of an initial Level Set. We could have used the
  //  \doxygen{DanielssonDistanceMapImageFilter} in the same way.
  //
  //  The seeds are passed stored in a container. The type of this
  //  container is defined as \code{NodeContainer} among the
  //  FastMarchingImageFilter traits.
  //
  
	 FastMarchingFilterType::NodeType::IndexType index;  
	  index[0] = properties_->seedX;
	  index[1] = properties_->seedY;
	  index[2] = properties_->seedZ;
	  _seeds->ElementAt(0).SetIndex(index);
	  _seeds->ElementAt(0).SetValue(- properties_->initialDistance);
  
  
//  initSeedNode_ = &seeds->ElementAt(0);

  fastMarching->SetTrialPoints(_seeds);

  //  Since the FastMarchingImageFilter is used here just as a
  //  Distance Map generator. It does not require a speed image as input.
  //  Instead the constant value $1.0$ is passed using the
  //  SetSpeedConstant() method.  
  fastMarching->SetSpeedConstant( 1.0 );
  
  fastMarching->SetOutputSize(
  			this->GetInputITKImage()->GetLargestPossibleRegion().GetSize());
  
  // set OutputITKImage as output for fast marching (FM) 
  fastMarching->GraftOutput(_levelSetImage);
  
  // let the FM to generate data in to OutputITKImage
  std::cout << "Running fast marching filter ..." << std::endl;
  fastMarching->Update();
  std::cout << "done" << std::endl;
  
  // now output of the FM should be in OutputITKImage
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::PrepareOutputDatasets(void)
{
	typename PredecessorType::ITKOutputImageType::RegionType region;
	typename PredecessorType::ITKOutputImageType::SpacingType spacing;
	
	ITKIntegration::ConvertMedevedImagePropsToITKImageProps<
		InputImageType, typename PredecessorType::ITKOutputImageType>(
				region, spacing, *this->in);
	
	SetOutImageSize(region, spacing);
	PredecessorType::PrepareOutputDatasets();
	
	AlocateAlignedImageData(region.GetSize());
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
::AlocateAlignedImageData(const typename ITKOutputImageType::SizeType &size)
{	
	_levelSetImage->CopyInformation(this->GetInputITKImage());
	
	// free old
	if(_levelSetImageData) free(_levelSetImageData);
	
	try {
	size_t sizeOfData = 1;	// size in elements (not in bytes)
	// count num of elems
	for( uint32 i=0; i< InputImageType::Dimension; i++)
		sizeOfData *= size[i];
	
	typedef union {
		TLSImaPixel **sp;
		void **vp;
	} UTLSImaPixelToInt;
	
	UTLSImaPixelToInt uConv; uConv.sp = &_levelSetImageData;
	// alocate new (aligned buffer)
	if( posix_memalign(uConv.vp, 128,
			sizeOfData * sizeof(TLSImaPixel) ) != 0)
	{
		throw std::bad_alloc();
	}
	_levelSetImage->GetPixelContainer()->SetImportPointer(
			_levelSetImageData,
				(typename TLevelSetImage::PixelContainer::ElementIdentifier) sizeOfData,
				false);
	} catch(...) {
		if(_levelSetImageData) free(_levelSetImageData);
		
		throw;
	}
	
	_levelSetImage->SetRegions(_levelSetImage->GetLargestPossibleRegion());
	_levelSetImage->Allocate();
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::SetupBinaryThresholder(void)
{
  thresholder = ThresholdingFilterType::New();
  thresholder->SetInput(_levelSetImage);
		                        
  thresholder->SetLowerThreshold( -1000.0 );
  thresholder->SetUpperThreshold(     0.0 );

  thresholder->SetOutsideValue(  0  );
  thresholder->SetInsideValue(  255 );
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::RunLevelSetSegmentator(void)
{
  typename ThresholdSegmentationFilterType::Pointer thresholdSegmentation = 
	  			ThresholdSegmentationFilterType::New();  
  
  ApplyProperties(thresholdSegmentation);
  
  // and set some properties	  
  thresholdSegmentation->SetMaximumRMSError(0.02);  		
  
  thresholdSegmentation->SetFeatureImage( this->GetInputITKImage() );
  thresholdSegmentation->SetInput( _levelSetImage );
  thresholdSegmentation->GraftOutput(_levelSetImage); 
  
  thresholdSegmentation->Update();
  
  thresholdSegmentation->PrintStats(std::cout);
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::ApplyProperties(
			typename ThresholdSegmentationFilterType::Pointer &thresholdSegmentation)
{
  thresholdSegmentation->SetNumberOfIterations( properties_->maxIterations );
  thresholdSegmentation->SetUpperThreshold( properties_->upperThreshold );
  thresholdSegmentation->SetLowerThreshold( properties_->lowerThreshold );
  thresholdSegmentation->SetPropagationScaling( properties_->propagationScaling );
  //thresholdSegmentation->GetDiffFunction()->SetAdvectionScaling( properties_->advectionScaling);
  thresholdSegmentation->SetCurvatureScaling( properties_->curvatureScaling);
  
  thresholdSegmentation->Modified();
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
bool
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>::CheckRun()
{
	typename ITKInputImageType::RegionType::SizeType size = 
		this->GetInputITKImage()->GetLargestPossibleRegion().GetSize();
	bool good = true;
	
	// check dataset
	if(size[0] < 32 || size[1] < 32
			|| (size[0] % 32) != 0 || (size[1] % 32) != 0)
	{
		std::cout << "Wrong dataset size!" << std::endl;
		good = false;
	}
			
	if( (properties_->seedX < 0 || properties_->seedX >= size[0])
			|| (properties_->seedY < 0 || properties_->seedY >= size[1])
			|| (properties_->seedZ < 0 || properties_->seedZ >= size[2]) )
	{
		std::cout << "Wrong seed!" << std::endl;
		good = false;
	}
	if(properties_->maxIterations < 1)
	{
		std::cout << "Wrong maxIterations!" << std::endl;
		good = false;	
	}
	if(properties_->initialDistance < 1 || properties_->initialDistance > (size[0] / 2))
	{
		std::cout << "Wrong initial size!" << std::endl;
		good = false;
	}
	
	return good;
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
bool
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::ProcessImage(const InputImageType &in, OutputImageType &out)
{
	PrintRunInfo(std::cout);
	
	if(! CheckRun())
			return false;
	try {
		RunFastMarchingFilter();		
		RunLevelSetSegmentator();
		
		  // berform binary thresholding to distinguish inner and outer part of LS
		//  		thresholder->SetInput(_levelSetImage);		
		 
		thresholder->Modified();	// to force recalculation 
		thresholder->Update();
		 
	} catch(itk::ExceptionObject &ex) {
		LOUT << ex << std::endl;
		std::cerr << ex << std::endl;
		return false;
	} catch(...) {
		LOG("exception thrown during Medved filter exec, returning false");
		std::cout << "exception thrown during Medved filter exec" << std::endl;
	}
	
	// separate particular runs
	std::cout << "############################################################"
	<< std::endl;
	
	return true;
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::PrintRunInfo(std::ostream &stream)
{
	stream << "==============================================" << std::endl;
	stream << "Dataset info:" << std::endl;
	stream << "size: " 
		<< this->GetInputITKImage()->GetLargestPossibleRegion().GetSize() 
		<< std::endl;
	stream << "==============================================" << std::endl;
	stream << "Filter started with these values:" << std::endl;
	stream << "Seed: " << properties_->seedX << ", " << properties_->seedY << ", " << properties_->seedZ << std::endl;
	stream << "Init distance: " << properties_->initialDistance << std::endl;
	stream << "Threshold: <" << properties_->lowerThreshold << "," << properties_->upperThreshold << ">" << std::endl;
	stream << "Max iteration: " << properties_->maxIterations << std::endl;
	stream << "Speed scaling: " << properties_->propagationScaling << std::endl;
	stream << "Curvature scaling: " << properties_->curvatureScaling << std::endl;
	stream << "==============================================" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////

#endif

