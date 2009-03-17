#ifndef DECIMATORFILTER_H_
#define DECIMATORFILTER_H_

#include "itkIntegration/itkFilter.h"
#include "Imaging/Image.h"
#include "itkImage.h"

#include "itkResampleImageFilter.h"
#include "itkSimilarity3DTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"


namespace M4D
{
namespace RemoteComputing
{

template< typename InputElementType, typename OutputElementType >
class DecimatorFilter
	: public ITKIntegration::ITKFilter< Imaging::Image<InputElementType, 3>, Imaging::Image<OutputElementType, 3> >
{
public:
	typedef ITKIntegration::ITKFilter< Imaging::Image<InputElementType, 3>, Imaging::Image<OutputElementType, 3> >
		PredecessorType;	
	typedef Imaging::Image<InputElementType, 3> InputImageType;
	typedef Imaging::Image<OutputElementType, 3> OutputImageType;
	
	typedef itk::Image<InputElementType, 3> ITKInputImageType;
	typedef itk::Image<OutputElementType, 3> ITKOutputImageType;
	
	static const uint32 Dim = 3;
	
	DecimatorFilter(float32 ratio);
	
	void PrepareOutputDatasets(void);
	
protected:
	bool ProcessImage(
				const InputImageType 	&in,
				OutputImageType		&out
			    );
	
private:
	float32 m_ratio;
	
	typedef itk::ResampleImageFilter<ITKInputImageType, ITKOutputImageType> FilterType;
	typename FilterType::Pointer m_filter;
	
	typedef itk::Similarity3DTransform< double >  TransformType;
	TransformType::Pointer m_transform;
	
	typedef itk::NearestNeighborInterpolateImageFunction< ITKInputImageType, double >  InterpolatorType;
    typename InterpolatorType::Pointer m_interpolator;

	void PrintRunInfo(std::ostream &stream);
};

}
}

//include implementation
#include "src/decimatorFilter.cxx"

#endif /*SERVERLEVELSETSEGMENTATION_H_*/
