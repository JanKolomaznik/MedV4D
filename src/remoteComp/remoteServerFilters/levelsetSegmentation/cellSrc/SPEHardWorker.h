#ifndef SPEHARDWORKER_H_
#define SPEHARDWORKER_H_

#include "diffFunc.h"

namespace itk {

template<typename >
class SPEHardWorker
{
public:
	
	typename SegmentationFunctionType::Pointer func_;
	
	void ProcessStep(linkedList begin, LinkedList end);
	
	/**This function allows a subclass to override the way in which updates to
		   * output values are applied during each iteration.  The default simply
		   * follows the standard finite difference scheme of scaling the change by the
		   * timestep and adding to the value of the previous iteration. */
		  inline virtual ValueType CalculateUpdateValue(
		    const IndexType &itkNotUsed(idx),
		    const TimeStepType &dt,
		    const ValueType &value,
		    const ValueType &change)
		    { return (value + dt * change); }
		  
	  /** Container type used to store updates to the active layer. */
	  typedef std::vector<ValueType> UpdateBufferType;
		  
  /** The update buffer used to store change values computed in
  	   *  CalculateChange. */
  	  UpdateBufferType m_UpdateBuffer;
	
	  /** Adjusts the values in a single layer "to" using values in a neighboring
	   *  layer "from".  The list of indicies in "to" are traversed and assigned
	   *  new values appropriately. Any indicies in "to" without neighbors in
	   *  "from" are moved into the "promote" layer (or deleted if "promote" is
	   *  greater than the number of layers). "InOrOut" == 1 indicates this
	   *  propagation is inwards (more negative).  "InOrOut" == 2 indicates this
	   *  propagation is outwards (more positive). */   
	  void PropagateLayerValues(StatusType from, StatusType to,
	                            StatusType promote, int InOrOut);

	  /** Adjusts the values associated with all the index layers of the sparse
	   * field by propagating out one layer at a time from the active set. This
	   * method also takes care of deleting nodes from the layers which have been
	   * marked in the status image as having been moved to other layers. */
	  void PropagateAllLayerValues();

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
	  
	  /** Reserves memory in the update buffer. Called before each iteration. */
	  void AllocateUpdateBuffer();
	  
	  /** Multiplicative identity of the ValueType. */
	  static ValueType m_ValueOne;

	  /** Additive identity of the ValueType. */
	  static ValueType m_ValueZero;

	  /** Special status value which indicates pending change to another sparse
	   *  field layer. */
	  static StatusType m_StatusChanging;

	  /** Special status value which indicates a pending change to a more positive
	   *  sparse field. */
	  static StatusType m_StatusActiveChangingUp;

	  /** Special status value which indicates a pending change to a more negative
	   *  sparse field. */
	  static StatusType m_StatusActiveChangingDown;

	  /** Special status value which indicates a pixel is on the boundary of the
	   *  image */
	  static StatusType m_StatusBoundaryPixel;

	  /** Special status value used as a default for indicies which have no
	      meaningful status. */
	  static StatusType m_StatusNull;	
};

}

//include implementation
#include "src/SPEHardWorker.tcc"

#endif /*HARDWORKER_H_*/
