
#ifndef SERVERLEVELSETSEGMENTATION_H_
#error File medevedWrapperFilter.cxx cannot be included directly!
#else

namespace M4D
{
namespace RemoteComputing
{
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::ThreshLSSegMedvedWrapper(Properties *props)
	: properties_(props)
	, initSeedNode_(NULL)
{
	// setup filters
	SetupBinaryThresholder();	
	SetupFastMarchingFilter();  
	SetupLevelSetSegmentator();
	
	featureToFloatCaster = FeatureToFloatFilterType::New();
	//floatToFeature = FloatToFeatureFilterType::New();
	 
	featureToFloatCaster->SetInput( this->GetInputITKImage() );	
	thresholdSegmentation->SetFeatureImage( featureToFloatCaster->GetOutput() );
	thresholdSegmentation->SetInput( fastMarching->GetOutput() );
	thresholder->SetInput( thresholdSegmentation->GetOutput() );
	//floatToFeature->SetInput(thresholdSegmentation->GetOutput());
	  
	// connect the pipeline into in/out of the ITKFilter
	//SetOutputITKImage( floatToFeature->GetOutput() );
	SetOutputITKImage( thresholder->GetOutput() );
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::SetupFastMarchingFilter(void)
{
  fastMarching = FastMarchingFilterType::New();
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
  typedef FastMarchingFilterType::NodeContainer           NodeContainer;

  // crete seed node as begining for fast marching filter		  
  NodeType node;
  
  NodeContainer::Pointer seeds = NodeContainer::New();
  seeds->Initialize();
  seeds->InsertElement( 0, node );
  
  initSeedNode_ = &seeds->ElementAt(0);

  fastMarching->SetTrialPoints(  seeds  );

  //  Since the FastMarchingImageFilter is used here just as a
  //  Distance Map generator. It does not require a speed image as input.
  //  Instead the constant value $1.0$ is passed using the
  //  SetSpeedConstant() method.  
  fastMarching->SetSpeedConstant( 1.0 );
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
	
#ifdef FOR_CELL
	AlocateAlignedImageData(region.GetSize());
#endif
	
	fastMarching->SetOutputSize(
			this->GetInputITKImage()->GetLargestPossibleRegion().GetSize());
}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
::AlocateAlignedImageData(const typename ITKOutputImageType::SizeType &size)
{
	size_t sizeOfData = 1;	// size in elements (not in bytes)
	// count num of elems
	for( uint32 i=0; i< InputImageType::Dimension; i++)
		sizeOfData *= size[i];
		
	// alocate aligned image data
	typedef typename FeatureToFloatFilterType::OutputImageType TFeatureConvImage;
	TFeatureConvImage *o = featureToFloatCaster->GetOutput();	
	
	// alocate new (aligned buffer)
	typedef typename FeatureToFloatFilterType::OutputImageType::PixelType TFeatureConvPixel;
	TFeatureConvPixel *dataPointer;
	if( posix_memalign((void**)(&dataPointer), 128,
			sizeOfData * sizeof(TFeatureConvPixel) ) != 0)
	{
		throw "bad";
	}
	o->GetPixelContainer()->SetImportPointer(
				dataPointer,
				sizeOfData,
				true);
	
	typedef typename ThresholdSegmentationFilterType::OutputImageType TSegmOutImage;
	TSegmOutImage *o2 = thresholdSegmentation->GetOutput();	
	
	// alocate new (aligned buffer)
	typedef typename ThresholdSegmentationFilterType::OutputImageType::PixelType TSegmOImPixel;
	TSegmOImPixel *data2Pointer;
	if( posix_memalign((void**)(&data2Pointer), 128,
			sizeOfData * sizeof(TSegmOImPixel) ) != 0)
	{
		throw "bad";
	}
	o2->GetPixelContainer()->SetImportPointer(
				data2Pointer,
				(typename ITKInputImageType::PixelContainer::ElementIdentifier) sizeOfData,
				true);
}
#endif
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::SetupBinaryThresholder(void)
{
  thresholder = ThresholdingFilterType::New();
		                        
  thresholder->SetLowerThreshold( -1000.0 );
  thresholder->SetUpperThreshold(     0.0 );

  thresholder->SetOutsideValue(  0  );
  thresholder->SetInsideValue(  255 );
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::SetupLevelSetSegmentator(void)
{
  thresholdSegmentation = ThresholdSegmentationFilterType::New();  
  // and set some properties	  
  thresholdSegmentation->SetMaximumRMSError( 0.02 );  
  thresholdSegmentation->SetIsoSurfaceValue(0.0);
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::ApplyProperties(void)
{
  thresholdSegmentation->SetNumberOfIterations( properties_->maxIterations );
  thresholdSegmentation->SetUpperThreshold( properties_->upperThreshold );
  thresholdSegmentation->SetLowerThreshold( properties_->lowerThreshold );
  thresholdSegmentation->SetPropagationScaling( properties_->propagationScaling );
  //thresholdSegmentation->GetDiffFunction()->SetAdvectionScaling( properties_->advectionScaling);
  thresholdSegmentation->SetCurvatureScaling( properties_->curvatureScaling);
  
  NodeType::IndexType index;  
  index[0] = properties_->seedX;
  index[1] = properties_->seedY;
  index[2] = properties_->seedZ;
  initSeedNode_->SetIndex(index);
  initSeedNode_->SetValue(- properties_->initialDistance);
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
bool
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::ProcessImage(const InputImageType &in, OutputImageType &out)
{
	ApplyProperties();
	try {
		PrintRunInfo(std::cout);
		thresholder->ResetPipeline();
		thresholder->Update();		
		thresholdSegmentation->PrintStats(std::cout);
	} catch (itk::ExceptionObject &ex) {
		LOUT << ex << std::endl;
		std::cerr << ex << std::endl;
		return false;
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ThreshLSSegMedvedWrapper<InputElementType, OutputElementType>
	::PrintRunInfo(std::ostream &stream)
{
	stream << "Filter started with these values:" << std::endl;
	stream << "Seed: " << properties_->seedX << ", " << properties_->seedY << ", " << properties_->seedZ << std::endl;
	stream << "Init distance: " << properties_->initialDistance << std::endl;
	stream << "Threshold: <" << properties_->lowerThreshold << "," << properties_->upperThreshold << ">" << std::endl;
	stream << "Max iteration: " << properties_->maxIterations << std::endl;
	stream << "Speed scaling: " << properties_->propagationScaling << std::endl;
	stream << "Curvature scaling: " << properties_->curvatureScaling << std::endl;
	
	//fastMarching->PrintSelf(stream, NULL);
	//thresholdSegmentation->PrintSelf(stream, NULL);
}

///////////////////////////////////////////////////////////////////////////////
}
}

#endif

