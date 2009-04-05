#ifndef GLOBALDATA_H_
#define GLOBALDATA_H_

#include "vnl/vnl_matrix_fixed.h"

namespace M4D {
namespace Cell {

template< typename ScalarValueType, uint16 Dim > 
struct GlobalDataStruct {
	ScalarValueType m_MaxAdvectionChange;
	ScalarValueType m_MaxPropagationChange;
	ScalarValueType m_MaxCurvatureChange;

	/** Hessian matrix */
	vnl_matrix_fixed<ScalarValueType, Dim, Dim> m_dxy;

	/** Array of first derivatives*/
	ScalarValueType m_dx[Dim];

	ScalarValueType m_dx_forward[Dim];
	ScalarValueType m_dx_backward[Dim];

	ScalarValueType m_GradMagSqr;
};

}}

#endif /*GLOBALDATA_H_*/
