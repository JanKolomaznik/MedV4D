#include "segmentator.h"

///////////////////////////////////////////////////////////////////////////////

Segmentator::Segmentator(
		float upperThreshold, float lowerThreshold, float curvatureScaling,
		float seedX, float seedY, float seedZ, double initialDistance,
		const ReadWriteImageType::SizeType &imageSize)
{ 
	// setup filters
	SetupBinaryThresholder();	
	SetupFastMarchingFilter(seedX, seedY, seedZ, initialDistance, imageSize);  
	SetupLevelSetSegmentator(upperThreshold, lowerThreshold, curvatureScaling);    
 
  // connect the filters ...
  thresholdSegmentation->SetInput( fastMarching->GetOutput() );  
  thresholder->SetInput( thresholdSegmentation->GetOutput() );
}

///////////////////////////////////////////////////////////////////////////////

void Segmentator::PrintInfo(void)
{
	// Print out some useful information 
	  std::cout << std::endl;
	  std::cout << "Max. no. iterations: " << thresholdSegmentation->GetNumberOfIterations() << std::endl;
	  std::cout << "Max. RMS error: " << thresholdSegmentation->GetMaximumRMSError() << std::endl;
	  std::cout << std::endl;
	  std::cout << "No. elpased iterations: " << thresholdSegmentation->GetElapsedIterations() << std::endl;
	  std::cout << "RMS change: " << thresholdSegmentation->GetRMSChange() << std::endl;
}

///////////////////////////////////////////////////////////////////////////////

void Segmentator::SetupBinaryThresholder(void)
{
	thresholder = ThresholdingFilterType::New();
	                        
	  thresholder->SetLowerThreshold( -1000.0 );
	  thresholder->SetUpperThreshold(     0.0 );

	  thresholder->SetOutsideValue(  0  );
	  thresholder->SetInsideValue(  255 );
}

///////////////////////////////////////////////////////////////////////////////

void Segmentator::SetupLevelSetSegmentator(
		float upperThreshold, float lowerThreshold, float curvatureScaling)
{
	thresholdSegmentation = ThresholdSegmentationLevelSetImageFilterType::New();
	  // and set some properties
	  thresholdSegmentation->SetPropagationScaling( 1.0 );
	  thresholdSegmentation->SetCurvatureScaling( curvatureScaling);
	  thresholdSegmentation->SetMaximumRMSError( 0.02 );
	  thresholdSegmentation->SetNumberOfIterations( 1200 );
	  thresholdSegmentation->SetUpperThreshold( upperThreshold );
	  thresholdSegmentation->SetLowerThreshold( lowerThreshold );
	  thresholdSegmentation->SetIsoSurfaceValue(0.0);
}

///////////////////////////////////////////////////////////////////////////////

void Segmentator::SetupFastMarchingFilter(
		float seedX, float seedY, float seedZ, 
		double initialDistance, const ReadWriteImageType::SizeType &imageSize)
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
	  typedef FastMarchingFilterType::NodeType                NodeType;

	  // crete seed node as begining for fast marching filter
	  InternalImageType::IndexType  seedPosition;	  
	  seedPosition[0] = seedX;
	  seedPosition[1] = seedY;
	  seedPosition[2] = seedZ;
	  
	  NodeType node;
	  const double seedValue = - initialDistance;
	  
	  node.SetValue( seedValue );
	  node.SetIndex( seedPosition );

	  NodeContainer::Pointer seeds = NodeContainer::New();
	  seeds->Initialize();
	  seeds->InsertElement( 0, node );

	  fastMarching->SetTrialPoints(  seeds  );

	  //  Since the FastMarchingImageFilter is used here just as a
	  //  Distance Map generator. It does not require a speed image as input.
	  //  Instead the constant value $1.0$ is passed using the
	  //  SetSpeedConstant() method.  
	  fastMarching->SetSpeedConstant( 1.0 );
	  
	  fastMarching->SetOutputSize( imageSize);
}

///////////////////////////////////////////////////////////////////////////////

//WriteInternalImages()
//{
//	  // We write out some intermediate images for debugging.  These images can
//	  // help tune parameters.
//	  //
//	  typedef itk::ImageFileWriter< InternalImageType > InternalWriterType;
//
//	  InternalWriterType::Pointer mapWriter = InternalWriterType::New();
//	  mapWriter->SetInput( fastMarching->GetOutput() );
//	  mapWriter->SetFileName("fastMarchingImage.mha");
//	  mapWriter->Update();
//
//	  InternalWriterType::Pointer speedWriter = InternalWriterType::New();
//	  speedWriter->SetInput( thresholdSegmentation->GetSpeedImage() );
//	  speedWriter->SetFileName("speedTermImage.mha");
//	  speedWriter->Update();
//}

///////////////////////////////////////////////////////////////////////////////