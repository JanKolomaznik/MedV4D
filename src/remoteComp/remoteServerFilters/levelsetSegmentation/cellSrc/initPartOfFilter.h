#ifndef INITPARTOFFILTER_H_
#define INITPARTOFFILTER_H_

#include "myFiniteDifferenceFilter.h"
//#include "itkThresholdSegmentationLevelSetImageFilter.h"

#include "supportClasses.h"
#include "common/perfCounter.h"
#include "itkSparseFieldLayer.h"
#include "itkObjectStore.h"
#include "itkNeighborhoodIterator.h"

#if( defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) )
	#include "PPE/SPEManager.h"
#define CELL
#else
#define PC
	#include "SPE/updateCalculation/updateCalculatorSPE.h"
	#include "SPE/applyUpdateCalc/applyUpdateCalculator.h"
#endif


namespace itk
{

template <class TInputImage, class TFeatureImage, class TOutputPixelType = float >
class MySegmtLevelSetFilter_InitPart	
	: public itk::MyFiniteDifferenceImageFilter<TInputImage, itk::Image<TOutputPixelType, TInputImage::ImageDimension> >
{
public:
	typedef MyFiniteDifferenceImageFilter<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> > Superclass;	
	typedef MySegmtLevelSetFilter_InitPart Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef typename TFeatureImage::PixelType FeaturePixelType;
	typedef Image<TOutputPixelType, TInputImage::ImageDimension> OutputImageType;
	  typedef typename OutputImageType::ValueType ValueType;
	
	typedef typename Superclass::TimeStepType TimeStepType;
	typedef typename Superclass::StatusType StatusType;
	
	/////////////////
	
	/** The data type used in numerical computations.  Derived from the output image type. */

  typedef typename OutputImageType::IndexType IndexType;

  /** Node type used in sparse field layer lists. */
  typedef SparseFieldLevelSetNode<IndexType> LayerNodeType;
  
  /** A list type used in the algorithm. */
  typedef SparseFieldLayer<LayerNodeType> LayerType;
  typedef typename LayerType::Pointer     LayerPointerType;
  
  typedef typename Superclass::NeighborhoodScalesType NeighborhoodScalesType;

  /** A type for a list of LayerPointerTypes */
  typedef std::vector<LayerPointerType> LayerListType;
  
  /** The type of the image used to index status information.  Necessary for
   *  the internals of the algorithm. */
  typedef Image<StatusType, OutputImageType::ImageDimension>  StatusImageType;

  /** Memory pre-allocator used to manage layer nodes in a multi-threaded
   *  environment. */
  typedef ObjectStore<LayerNodeType> LayerNodeStorageType;

  /** Container type used to store updates to the active layer. */
  typedef std::vector<ValueType> UpdateBufferType;
  
  //////////////
  
	void SetUpperThreshold(FeaturePixelType upThreshold) { m_Conf.m_upThreshold = upThreshold; }
	void SetLowerThreshold(FeaturePixelType loThreshold) { m_Conf.m_downThreshold = loThreshold; }
	void SetPropagationWeight(float32 propWeight) { m_Conf.m_propWeight = propWeight; }
	void SetCurvatureWeight(float32 curvWeight) { m_Conf.m_curvWeight = curvWeight; }
	
	void SetIsoSurfaceValue(ValueType val) { m_IsoSurfaceValue = val; }
	
	void SetFeatureImage(const TFeatureImage *f)
	  {
	    this->ProcessObject::SetNthInput( 1, const_cast< TFeatureImage * >(f) );
	  }
	
	TFeatureImage * GetFeatureImage()
	  	  { return ( static_cast< TFeatureImage *>(this->ProcessObject::GetInput(1)) ); }
	
	void PrintStats(std::ostream &s);
	
	void PrintUpdateBuf(std::ostream &s);
	
	// **************************************

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

	  /** An array which contains all of the layers needed in the sparse
	   * field. Layers are organized as follows: m_Layer[0] = active layer, 
	   * m_Layer[i:odd] = inside layer (i+1)/2, m_Layer[i:even] = outside layer i/2
	  */
	  LayerListType m_Layers;

	  /** The number of layers to use in the sparse field.  Sparse field will
	   * consist of m_NumberOfLayers layers on both sides of a single active layer.
	   * This active layer is the interface of interest, i.e. the zero
	   * level set. */
	  unsigned int m_NumberOfLayers;

	  /** An image of status values used internally by the algorithm. */
	  typename StatusImageType::Pointer m_StatusImage;

	  /** Storage for layer node objects. */
	  typename LayerNodeStorageType::Pointer m_LayerNodeStore;
	  
	  /** The value in the input which represents the isosurface of interest. */
	  ValueType m_IsoSurfaceValue;

	  /** The update buffer used to store change values computed in
	   *  CalculateChange. */
	  UpdateBufferType m_UpdateBuffer;
	  
	  bool m_BoundsCheckingActive;
	  
	  typedef NeighborhoodIterator<OutputImageType> NeighbourIterT;
	  
	  /** Connectivity information for examining neighbor pixels.   */
	    SparseFieldCityBlockNeighborList< typename NeighbourIterT::RadiusType, typename NeighbourIterT::OffsetType, 3 >
	    m_NeighborList;
	    
	    /** The constant gradient to maintain between isosurfaces in the
	        sparse-field of the level-set image.  This value defaults to 1.0 */
	    double m_ConstantGradientValue;
	    
protected:
	MySegmtLevelSetFilter_InitPart(void);
	~MySegmtLevelSetFilter_InitPart(void);
	
    typedef M4D::Cell::RunConfiguration TRunConf;
    TRunConf m_Conf;
	
private:
	PerfCounter cntr_;
};

}
//include implementation
#include "src/initPartOfFilter.tcc"

#endif /*INITPARTOFFILTER_H_*/
