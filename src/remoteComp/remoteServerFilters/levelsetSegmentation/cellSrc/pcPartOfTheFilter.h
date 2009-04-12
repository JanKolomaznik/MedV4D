#ifndef PCPARTOFTHEFILTER_H_
#define PCPARTOFTHEFILTER_H_

#include "initPartOfFilter.h"

#include "supportClasses.h"
#include "common/perfCounter.h"
#include "itkSparseFieldLayer.h"
#include "itkObjectStore.h"
#include "itkNeighborhoodIterator.h"

namespace itk
{

template <class TInputImage, class TFeatureImage, class TOutputPixelType = float >
class PCPartOfSegmtLevelSetFilter
	: public itk::MySegmtLevelSetFilter_InitPart<TInputImage, itk::Image<TOutputPixelType, TInputImage::ImageDimension> >
{
public:
	typedef MySegmtLevelSetFilter_InitPart<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> > Superclass;	
	typedef PCPartOfSegmtLevelSetFilter Self;
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
	
	// **************************************

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
//	    void PropagateLayerValues(StatusType from, StatusType to,
//	                              StatusType promote, int InOrOut);
	    
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
	    
protected:
	PCPartOfSegmtLevelSetFilter(void);
	~PCPartOfSegmtLevelSetFilter(void);
	
	void InitConfigStructures(void);
	
	typedef M4D::Cell::UpdateCalculatorSPE TUpdateCalculatorSPE;
	TUpdateCalculatorSPE updateSolver;
	
	M4D::Cell::ApplyUpdateSPE applyUpdateCalc;
	
	M4D::Cell::LayerGate::LayerType *m_gateLayerPointers[LYERCOUNT];
	
	void SetupGate();
	
private:
	PerfCounter cntr_;
};

}
//include implementation
#include "src/pcPartOfTheFilter.tcc"

#endif /*PCPARTOFTHEFILTER_H_*/
