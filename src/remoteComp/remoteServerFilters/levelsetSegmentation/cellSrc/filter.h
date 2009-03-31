#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#define CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_

#include "myFiniteDifferenceImageFilter.h"
#include "supportClasses.h"
#include "itkMultiThreader.h"
#include "itkSparseFieldLayer.h"
#include "itkObjectStore.h"
#include <vector>
#include "itkNeighborhoodIterator.h"

namespace itk
{

template <class TInputImage, class TFeatureImage, class TOutputPixelType = float >
class ThreshSegLevelSetFilter	
	: public MyFiniteDifferenceImageFilter<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> >
{
public:
	/** Standard class typedefs */
	typedef Image<TOutputPixelType, TInputImage::ImageDimension> OutputImageType;
	typedef TInputImage                         InputImageType;
	typedef TFeatureImage                       FeatureImageType;
	
	typedef ThreshSegLevelSetFilter<InputImageType, FeatureImageType, TOutputPixelType> Self;
	  typedef MyFiniteDifferenceImageFilter<InputImageType, OutputImageType> Superclass;
	  typedef SmartPointer<Self>                                     Pointer;
	  typedef SmartPointer<const Self>                               ConstPointer;

	  /**Typedefs from the superclass */
	  typedef typename Superclass::TimeStepType           TimeStepType;
	  typedef typename Superclass::RadiusType             RadiusType;
	  typedef typename Superclass::NeighborhoodScalesType NeighborhoodScalesType;
	  
	  /** Method for creation through the object factory. */
	  itkNewMacro(Self);

	  /** Run-time type information (and related methods). */
	  itkTypeMacro(ThreshSegLevelSetFilter, MyFiniteDifferenceImageFilter);

	  /** Information derived from the image types. */
	  
	  typedef typename OutputImageType::IndexType IndexType;
	  itkStaticConstMacro(ImageDimension, unsigned int,
	                      TOutputImage::ImageDimension);

	  /** The data type used in numerical computations.  Derived from the output
	   *  image type. */
	  typedef typename OutputImageType::ValueType ValueType;

	  /** Node type used in sparse field layer lists. */
	  typedef SparseFieldLevelSetNode<IndexType> LayerNodeType;
	  
	  /** A list type used in the algorithm. */
	  typedef SparseFieldLayer<LayerNodeType> LayerType;
	  typedef typename LayerType::Pointer     LayerPointerType;

	  /** A type for a list of LayerPointerTypes */
	  typedef std::vector<LayerPointerType> LayerListType;
	  
	  /** Type used for storing status information */
	  typedef signed char StatusType;
	  
	  /** The type of the image used to index status information.  Necessary for
	   *  the internals of the algorithm. */
	  typedef Image<StatusType, itkGetStaticConstMacro(ImageDimension)>
	  StatusImageType;

	  /** Memory pre-allocator used to manage layer nodes in a multi-threaded
	   *  environment. */
	  typedef ObjectStore<LayerNodeType> LayerNodeStorageType;

	  /** Set/Get the number of layers to use in the sparse field.  Argument is the
	   *  number of layers on ONE side of the active layer, so the total layers in
	   *   the sparse field is 2 * NumberOfLayers +1 */
	  itkSetMacro(NumberOfLayers, unsigned int);
	  itkGetMacro(NumberOfLayers, unsigned int);

	  /** Set/Get the value of the isosurface to use in the input image. */
	  itkSetMacro(IsoSurfaceValue, ValueType);
	  itkGetMacro(IsoSurfaceValue, ValueType);

	  /** Get the RMS change calculated in the PREVIOUS iteration.  This value is
	   *  the square root of the average square of the change value of all pixels
	   *  updated during the previous iteration. */
	  //  itkGetMacro(RMSChange, ValueType);
	  
	void SetUpperThreshold(FeaturePixelType upThreshold) {}
  	void SetLowerThreshold(FeaturePixelType upThreshold) {}
  	void SetPropagationWeight(float32 propWeight) {}
  	void SetCurvatureWeight(float32 curvWeight) {}
	  	
  	void SetFeatureImage(const TFeatureImage *f)
  	  {
  	    this->ProcessObject::SetNthInput( 1, const_cast< TFeatureImage * >(f) );
  	    func_->SetFeatureImage(f);
  	  }
  	
  	void PrintStats(std::ostream &s);

protected:
	ThreshSegLevelSetFilter(void);
	~ThreshSegLevelSetFilter(void) {}	  

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
	  void Initialize();

	  /** Copies the input to the output image.  Processing occurs on the output
	   * image, so the data type of the output image determines the precision of
	   * the calculations (i.e. double or float).  This method overrides the
	   * parent class method to do some additional processing. */
	  void CopyInputToOutput();

	  /** Applies the update buffer values to the active layer and reconstructs the
	   *  sparse field layers for the next iteration. */
	  void ApplyUpdate(TimeStepType dt);

	  /** Traverses the active layer list and calculates the change at these
	   *  indicies to be applied in the current iteration. */
	  TimeStepType CalculateChange();

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
	  
	  itkGetConstMacro(ValueZero, ValueType);
	  itkGetConstMacro(ValueOne, ValueType);

	  /** Connectivity information for examining neighbor pixels.   */
	  SparseFieldCityBlockNeighborList<NeighborhoodIterator<OutputImageType> > m_NeighborList;
	  
	  /** The constant gradient to maintain between isosurfaces in the
	      sparse-field of the level-set image.  This value defaults to 1.0 */
	  double m_ConstantGradientValue;
	    
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

	  

	  /** The RMS change calculated from each update.  Can be used by a subclass to
	   *  determine halting criteria.  Valid only for the previous iteration, not
	   *  during the current iteration.  Calculated in ApplyUpdate. */
	  //  ValueType m_RMSChange;

	private:
	  SparseFieldLevelSetImageFilter(const Self&);//purposely not implemented
	  void operator=(const Self&);      //purposely not implemented

	  /** This flag is true when methods need to check boundary conditions and
	      false when methods do not need to check for boundary conditions. */
	  bool m_BoundsCheckingActive;	  
	
	  FeaturePixelType m_upThreshold;
	  FeaturePixelType m_downThreshold;
	  float32 m_propWeight;
	  float32 m_curvWeight;
};

}
//include implementation
#include "src/filter.tcc"

#endif /*CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_*/
