#ifndef CONSTS_H_
#define CONSTS_H_

namespace itk {

template<typename ValueType>
class CommonBase
{
public:
	typedef double  TimeStepType;
	  
	/** Type used for storing status information */
	typedef signed char StatusType;
	
	/** Multiplicative identity of the ValueType. */
  ValueType m_ValueOne;//NumericTraits<ValueType>::One;

  /** Additive identity of the ValueType. */
  ValueType m_ValueZero; //NumericTraits<ValueType>::Zero;

  /** Special status value which indicates pending change to another sparse
   *  field layer. */
  StatusType m_StatusChanging;

  /** Special status value which indicates a pending change to a more positive
   *  sparse field. */
  StatusType m_StatusActiveChangingUp;

  /** Special status value which indicates a pending change to a more negative
   *  sparse field. */
  StatusType m_StatusActiveChangingDown;

  /** Special status value which indicates a pixel is on the boundary of the
   *  image */
  StatusType m_StatusBoundaryPixel;

  /** Special status value used as a default for indicies which have no
      meaningful status. */
  StatusType m_StatusNull;
  
protected:
	CommonBase()
	{
		m_ValueOne = NumericTraits<ValueType>::One;
		m_ValueZero = NumericTraits<ValueType>::Zero;
		 m_StatusChanging = -1;
		 m_StatusActiveChangingUp = -2;
		 m_StatusActiveChangingDown = -3;
		m_StatusBoundaryPixel = -4;
		m_StatusNull = NumericTraits<StatusType>::NonpositiveMin();
	}
};

}

#endif /*CONSTS_H_*/
