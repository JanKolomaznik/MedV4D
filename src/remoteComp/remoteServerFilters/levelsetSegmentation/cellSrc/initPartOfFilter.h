#ifndef INITPARTOFFILTER_H_
#define INITPARTOFFILTER_H_

#include "myFiniteDifferenceFilter.h"
#include "PPE/SPEManager.h"
#include "SPE/commonTypes.h"

#include "supportClasses.h"
#include "common/perfCounter.h"

#include "itkNeighborhoodIterator.h"
#include "itkZeroCrossingImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkShiftScaleImageFilter.h"
#include "itkNeighborhoodAlgorithm.h"

namespace M4D
{
namespace Cell
{

template <class TInputImage, class TFeatureImage, class TOutputPixelType =float > class MySegmtLevelSetFilter_InitPart :
	public MyFiniteDifferenceImageFilter<TInputImage, itk::Image<TOutputPixelType, TInputImage::ImageDimension> >
{
public:
	typedef MyFiniteDifferenceImageFilter<TInputImage, itk::Image<TOutputPixelType, TInputImage::ImageDimension> >
			Superclass;
	typedef MySegmtLevelSetFilter_InitPart Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef typename TFeatureImage::PixelType FeaturePixelType;
	typedef itk::Image<TOutputPixelType, TInputImage::ImageDimension>
			OutputImageType;
	typedef typename OutputImageType::ValueType ValueType;
	typedef typename OutputImageType::IndexType IndexType;

	//typedef WorkManager<TIndex, ValueType> TWorkManager;

	/** The type of the image used to index status information.  Necessary for
	 *  the internals of the algorithm. */
	typedef itk::Image<StatusType, OutputImageType::ImageDimension>
			StatusImageType;

	M4D::Cell::TIndex ToMyIndex(const IndexType &i);
	IndexType ToITKIndex(const M4D::Cell::TIndex &i);
	//////////////

	void SetUpperThreshold(FeaturePixelType upThreshold)
	{
		m_runConf.m_upThreshold = upThreshold;
	}
	void SetLowerThreshold(FeaturePixelType loThreshold)
	{
		m_runConf.m_downThreshold = loThreshold;
	}
	void SetPropagationScaling(float32 propWeight)
	{
		m_runConf.m_propWeight = propWeight;
	}
	void SetCurvatureScaling(float32 curvWeight)
	{
		m_runConf.m_curvWeight = curvWeight;
	}

	void SetIsoSurfaceValue(ValueType val)
	{
		m_IsoSurfaceValue = val;
	}

	void SetFeatureImage(const TFeatureImage *f)
	{
		this->itk::ProcessObject::SetNthInput( 1,
				const_cast< TFeatureImage * >(f) );
	}

	TFeatureImage * GetFeatureImage()
	{
		return ( static_cast< TFeatureImage *>(this->itk::ProcessObject::GetInput(1)) );
	}

	void PrintStats(std::ostream &s);

	// FUNCTIONS

	/**This method packages the output(s) into a consistent format.  The default
	 * implementation produces a volume with the final solution values in the
	 * sparse field, and inside and outside values elsewhere as appropriate. */
	virtual void PostProcessOutput();

	/**This method pre-processes pixels inside and outside the sparse field
	 * layers.  The default is to set them to positive and negative values,
	 * respectively. This is not necessary as part of the calculations, but
	 * produces a more intuitive output for the user. */
	virtual void InitializeBackgroundPixels();

	/** Constructs the sparse field layers and initializes their values. */
	void InitializeInputAndConstructLayers();

	/** Copies the input to the output image.  Processing occurs on the output
	 * image, so the data type of the output image determines the precision of
	 * the calculations (i.e. double or float).  This method overrides the
	 * parent class method to do some additional processing. */
	void CopyInputToOutput();

	/** Reserves memory in the update buffer. Called before each iteration. */
	void AllocateUpdateBuffer();

	/** Initializes a layer of the sparse field using a previously initialized
	 * layer. Builds the list of nodes in m_Layer[to] using m_Layer[from].
	 * Marks values in the m_StatusImage. */
	void ConstructLayer(StatusType from, StatusType to);

	/** Constructs the active layer and initialize the first layers inside and
	 * outside of the active layer.  The active layer defines the position of the
	 * zero level set by its values, which are constrained within a range around
	 *  zero. */
	void ConstructActiveLayer();

	/** Initializes the values of the active layer set. */
	void InitializeActiveLayerValues();

	void PropagateAllLayerValues();

	void InitConfigStructures(void);

	// MEMBERS
	/** This image is a copy of the input with m_IsoSurfaceValue subtracted from
	 * each pixel.  This way we only need to consider the zero level set in our
	 * calculations.  Makes the implementation easier and more efficient. */
	typename OutputImageType::Pointer m_ShiftedImage;

	/** An image of status values used internally by the algorithm. */
	typename StatusImageType::Pointer m_StatusImage;

	/** The value in the input which represents the isosurface of interest. */
	ValueType m_IsoSurfaceValue;

	bool m_BoundsCheckingActive;

	typedef itk::NeighborhoodIterator<OutputImageType> NeighbourIterT;

	/** Connectivity information for examining neighbor pixels.   */
	SparseFieldCityBlockNeighborList< typename NeighbourIterT::RadiusType, typename NeighbourIterT::OffsetType, 3>
			m_NeighborList;

	/** The constant gradient to maintain between isosurfaces in the
	 sparse-field of the level-set image.  This value defaults to 1.0 */
	double m_ConstantGradientValue;

protected:
	MySegmtLevelSetFilter_InitPart(void);
	~MySegmtLevelSetFilter_InitPart(void);

	void InitRunConf();

	RunConfiguration m_runConf __attribute__((aligned(RunConfiguration_Allign)));

	WorkManager _workManager;
	SPEManager m_SPEManager;
	
	typename StatusImageType::PixelType *_statusImageData;
	
private:
	PerfCounter cntr_;
};

//include implementation
#include "src/initPartOfFilter.tcc"

}
}

#endif /*INITPARTOFFILTER_H_*/
