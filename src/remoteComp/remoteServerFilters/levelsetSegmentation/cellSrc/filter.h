#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#define CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_


//#include "itkThresholdSegmentationLevelSetImageFilter.h"

#ifdef FOR_CELL
#include "PPE/SPEManager.h"
#include "initPartOfFilter.h"
#else	/* PC */
#include "common/Common.h"

#include "pcPartOfTheFilter.h"
#endif

namespace itk
{

template <class TInputImage, class TFeatureImage, class TOutputPixelType = float >
class MySegmtLevelSetFilter	
#ifdef FOR_CELL
	: public itk::MySegmtLevelSetFilter_InitPart<TInputImage, itk::Image<TOutputPixelType, TInputImage::ImageDimension> >
#else
	: public itk::PCPartOfSegmtLevelSetFilter<TInputImage, itk::Image<TOutputPixelType, TInputImage::ImageDimension> >
#endif
{
public:
	
	typedef MySegmtLevelSetFilter Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef typename TFeatureImage::PixelType FeaturePixelType;
	typedef Image<TOutputPixelType, TInputImage::ImageDimension> OutputImageType;
	  typedef typename OutputImageType::ValueType ValueType;
	
#ifdef FOR_CELL
	typedef MySegmtLevelSetFilter_InitPart<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> > Superclass;
#else
	typedef PCPartOfSegmtLevelSetFilter<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> > Superclass;
#endif
	
	typedef typename Superclass::TimeStepType TimeStepType;
	typedef typename Superclass::StatusType StatusType;
	
	itkNewMacro(Self);

	
	// **************************************

	  // FUNCTIONS
	  
	   
	    /** Constructs the sparse field layers and initializes their values. */
	    void Initialize();

	    /** Applies the update buffer values to the active layer and reconstructs the
	     *  sparse field layers for the next iteration. */
	    void ApplyUpdate(TimeStepType dt);

	    /** Traverses the active layer list and calculates the change at these
	     *  indicies to be applied in the current iteration. */
	    TimeStepType CalculateChange();
	    

	    /** Adjusts the values associated with all the index layers of the sparse
	     * field by propagating out one layer at a time from the active set. This
	     * method also takes care of deleting nodes from the layers which have been
	     * marked in the status image as having been moved to other layers. */
	    void PropagateAllLayerValues();
	    
protected:
	MySegmtLevelSetFilter(void);
	~MySegmtLevelSetFilter(void);
	
	void InitConfigStructures(void);
	
#ifdef FOR_CELL
	M4D::Cell::ESPUCommands command;
#endif
};

}
//include implementation
#include "src/filter.tcc"

#endif /*CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_*/
