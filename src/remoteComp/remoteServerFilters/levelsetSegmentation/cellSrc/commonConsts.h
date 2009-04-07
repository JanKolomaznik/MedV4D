#ifndef CONSTS_H_
#define CONSTS_H_

#include "SPE/commonTypes.h"

namespace M4D {
namespace Cell {

class Consts
{
public:
	
	/** Multiplicative identity of the ValueType. */
	TPixelValue m_ValueOne;//NumericTraits<ValueType>::One;

  /** Additive identity of the ValueType. */
	TPixelValue m_ValueZero; //NumericTraits<ValueType>::Zero;

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
	Consts()
	{
		m_ValueOne = 1;//itk::NumericTraits<ValueType>::One;
		m_ValueZero = 0;//itk::NumericTraits<ValueType>::Zero;
		 m_StatusChanging = -1;
		 m_StatusActiveChangingUp = -2;
		 m_StatusActiveChangingDown = -3;
		m_StatusBoundaryPixel = -4;
		m_StatusNull = -127;//itk::NumericTraits<StatusType>::NonpositiveMin();
	}
};

}}

#endif /*CONSTS_H_*/
