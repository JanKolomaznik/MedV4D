#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#define CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_

#include "initPartOfFilter.h"
//#include "itkThresholdSegmentationLevelSetImageFilter.h"

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
class MySegmtLevelSetFilter	
	: public itk::MySegmtLevelSetFilter_InitPart<TInputImage, itk::Image<TOutputPixelType, TInputImage::ImageDimension> >
{
public:
	typedef MySegmtLevelSetFilter_InitPart<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> > Superclass;	
	typedef MySegmtLevelSetFilter Self;
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
	
	itkNewMacro(Self);
	
	void SetUpperThreshold(FeaturePixelType upThreshold) { m_Conf.m_upThreshold = upThreshold; }
  	void SetLowerThreshold(FeaturePixelType loThreshold) { m_Conf.m_downThreshold = loThreshold; }
  	void SetPropagationWeight(float32 propWeight) { m_Conf.m_propWeight = propWeight; }
  	void SetCurvatureWeight(float32 curvWeight) { m_Conf.m_curvWeight = curvWeight; }
	

	
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
	    
	    /** Adjusts the values in a single layer "to" using values in a neighboring
	     *  layer "from".  The list of indicies in "to" are traversed and assigned
	     *  new values appropriately. Any indicies in "to" without neighbors in
	     *  "from" are moved into the "promote" layer (or deleted if "promote" is
	     *  greater than the number of layers). "InOrOut" == 1 indicates this
	     *  propagation is inwards (more negative).  "InOrOut" == 2 indicates this
	     *  propagation is outwards (more positive). */   
	    void PropagateLayerValues(StatusType from, StatusType to,
	                              StatusType promote, int InOrOut);
	    
	    ValueType CalculateUpdateValue(
	    		    const TimeStepType &dt,
	    		    const ValueType &value,
	    		    const ValueType &change)
	    		    {
	    			ValueType val = (value + dt * change); 
	    			return val;
	    			}


	    /** Updates the active layer values using m_UpdateBuffer. Also creates an
	     *  "up" and "down" list for promotion/demotion of indicies leaving the
	     *  active set. */
	    void UpdateActiveLayerValues(TimeStepType dt, LayerType *StatusUpList,
	                                 LayerType *StatusDownList);
	    /** */
	    void ProcessStatusList(LayerType *InputList, LayerType *OutputList,
	                           StatusType ChangeToStatus, StatusType SearchForStatus);

	    /** */
	    void ProcessOutsideList(LayerType *OutsideList, StatusType ChangeToStatus);
	
	  
	    
	    typedef M4D::Cell::RunConfiguration TRunConf;
	    TRunConf m_Conf;
	    
	    //M4D::Cell::ApplyUpdateConf m_applyUpdateConf;
	    
protected:
	MySegmtLevelSetFilter(void);
	~MySegmtLevelSetFilter(void);
	
	void InitConfigStructures(void);
	
#if( defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) )
	M4D::Cell::SPEManager m_SPEManager;
	M4D::Cell::ESPUCommands command;
#else
	typedef M4D::Cell::UpdateCalculatorSPE TUpdateCalculatorSPE;
	TUpdateCalculatorSPE updateSolver;
	
	M4D::Cell::ApplyUpdateSPE applyUpdateCalc;
	
	M4D::Cell::LayerGate::LayerType *m_gateLayerPointers[LYERCOUNT];
	
	void SetupGate();
#endif
	
	
	
private:
	PerfCounter cntr_;
};

}
//include implementation
#include "src/filter.tcc"

#endif /*CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_*/
