
#ifndef SERVERLEVELSETSEGMENTATION_H_
#error File serverLevelsetSegmentation.cxx cannot be included directly!
#else

namespace M4D
{
namespace RemoteComputing
{
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
ServerLevelsetSegmentation<InputElementType, OutputElementType>
	::ServerLevelsetSegmentation()
	: initSeedNode_(NULL)
{
	// setup filters
		SetupBinaryThresholder();	
		SetupFastMarchingFilter();  
		SetupLevelSetSegmentator();
		
		ApplyProperties();
	 
	  // connect the filters to form pipeline
	  thresholdSegmentation->SetInput( fastMarching->GetOutput() );  
	  thresholder->SetInput( thresholdSegmentation->GetOutput() );
	  
	  // connect the pipeline into in/out of the ITKFilter
	  thresholdSegmentation->SetFeatureImage( 
			  PredecessorType::GetInputITKImage() );
	  SetOutputITKImage( thresholder->GetOutput() );
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ServerLevelsetSegmentation<InputElementType, OutputElementType>
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
ServerLevelsetSegmentation<InputElementType, OutputElementType>
	::PrepareOutputDatasets(void)
{
	PredecessorType::PrepareOutputDatasets();

	const typename PredecessorType::ITKInputImageType *image = 
		PredecessorType::GetInputITKImage();
	fastMarching->SetOutputSize( image->GetLargestPossibleRegion().GetSize() );
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ServerLevelsetSegmentation<InputElementType, OutputElementType>
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
ServerLevelsetSegmentation<InputElementType, OutputElementType>
	::SetupLevelSetSegmentator(void)
{
  thresholdSegmentation = ThresholdSegmentationLevelSetImageFilterType::New();
  // and set some properties	  
  thresholdSegmentation->SetMaximumRMSError( 0.02 );  
  thresholdSegmentation->SetIsoSurfaceValue(0.0);
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
ServerLevelsetSegmentation<InputElementType, OutputElementType>
	::ApplyProperties(void)
{
  thresholdSegmentation->SetNumberOfIterations( properties_.maxIterations );
  thresholdSegmentation->SetUpperThreshold( properties_.upperThreshold );
  thresholdSegmentation->SetLowerThreshold( properties_.lowerThreshold );
  thresholdSegmentation->SetPropagationScaling( properties_.propagationScaling );
  thresholdSegmentation->SetAdvectionScaling( properties_.advectionScaling);
  thresholdSegmentation->SetCurvatureScaling( properties_.curvatureScaling);
  
  initSeedNode_->GetIndex()[0] = properties_.seedX;
  initSeedNode_->GetIndex()[1] = properties_.seedY;
  initSeedNode_->GetIndex()[2] = properties_.seedZ;
  initSeedNode_->SetValue(- properties_.initialDistance);
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
bool
ServerLevelsetSegmentation<InputElementType, OutputElementType>
	::ProcessImage(const InputImageType &in, OutputImageType &out)
{
	thresholdSegmentation->Update();
	
	return true;
}

///////////////////////////////////////////////////////////////////////////////
}
}

#endif

