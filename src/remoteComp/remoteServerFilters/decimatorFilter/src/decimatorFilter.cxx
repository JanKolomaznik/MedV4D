
#ifndef DECIMATORFILTER_H_
#error File decimatorFilter.cxx cannot be included directly!
#else

namespace M4D
{
namespace RemoteComputing
{
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
DecimatorFilter<InputElementType, OutputElementType>
::DecimatorFilter(float32 ratio)
	: m_ratio(ratio)
{
	m_filter = FilterType::New();
	  
	m_transform = TransformType::New();
	m_filter->SetTransform( m_transform );
	  
  m_interpolator = InterpolatorType::New();
  m_filter->SetInterpolator( m_interpolator );
  
  m_filter->SetDefaultPixelValue( 1000 );  
	    
	// connect the pipeline into in/out of the ITKFilter
	m_filter->SetInput( this->GetInputITKImage() );
	SetOutputITKImage( m_filter->GetOutput() );
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
void
DecimatorFilter<InputElementType, OutputElementType>
	::PrepareOutputDatasets(void)
{
	PredecessorType::PrepareOutputDatasets();
	
	typename PredecessorType::ITKOutputImageType::RegionType region;
	typename PredecessorType::ITKOutputImageType::SpacingType spacing;
	
	// get properties from input medved image
	ITKIntegration::ConvertMedevedImagePropsToITKImageProps<
		InputImageType, typename PredecessorType::ITKOutputImageType>(
				region, spacing, this->GetInputImage());
	
	typename PredecessorType::ITKOutputImageType::RegionType::SizeType size = region.GetSize();
	// multiply by the m_ratio except 3-rd coord
	for(uint32 i=0; i<Dim-1; i++)
	{
		spacing[i] *= m_ratio;
		size[i] *= m_ratio;
	}
	// 3- rd coord only in spacing
	spacing[Dim-1] *= m_ratio;
	region.SetSize(size);
	
	SetOutImageSize(region, spacing);
	
	m_transform->SetScale( m_ratio );
	
	typename ITKOutputImageType::PointType	origin;
	origin.Fill(0);
	
	m_filter->SetOutputSpacing( spacing );
	m_filter->SetOutputOrigin( origin );
	m_filter->SetSize( size );
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputElementType, typename OutputElementType >
bool
DecimatorFilter<InputElementType, OutputElementType>
	::ProcessImage(const InputImageType &in, OutputImageType &out)
{
	try {
		m_filter->Update();
		PrintRunInfo(std::cout);
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
DecimatorFilter<InputElementType, OutputElementType>
	::PrintRunInfo(std::ostream &stream)
{
	stream << "Run info: ratio = " << m_ratio << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
}
}

#endif

