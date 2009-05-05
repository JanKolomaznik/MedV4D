#ifndef GLOBALDATA_H_
#define GLOBALDATA_H_

//#include "vnl/vnl_matrix_fixed.h"
#include "../commonTypes.h"

namespace M4D
{
namespace Cell
{

struct GlobalDataStruct
{
	typedef TPixelValue ScalarValueType;
	
	GlobalDataStruct()
	{
		m_MaxAdvectionChange = m_MaxPropagationChange = m_MaxCurvatureChange = 0;
	}

	ScalarValueType m_MaxAdvectionChange;
	ScalarValueType m_MaxPropagationChange;
	ScalarValueType m_MaxCurvatureChange;

	/** Hessian matrix */
	ScalarValueType m_dxy[DIM][DIM];

	/** Array of first derivatives*/
	ScalarValueType m_dx[DIM];
	ScalarValueType m_dx_forward[DIM];
	ScalarValueType m_dx_backward[DIM];

	ScalarValueType m_GradMagSqr;

	void Print(std::ostream &s)
	{
		s << "Hessian" << std::endl;
		s << "m_GradMagSqr=" << m_GradMagSqr << std::endl;
		s << "m_dxy:" << std::endl;
		for (uint32 i=0; i<DIM; i++)
		{
			s << "[" << m_dxy[0][i] << "," << m_dxy[1][i] << "," << m_dxy[2][i]
					<< "]" << std::endl;
		}
		s << "m_dx:";
		s << "[" << m_dx[0] << "," << m_dx[1] << "," << m_dx[2] << "]"
				<< std::endl;
		s << "m_dx_forward:";
		s << "[" << m_dx_forward[0] << "," << m_dx_forward[1] << ","
				<< m_dx_forward[2] << "]" << std::endl;
		s << "m_dx_backward:";
		s << "[" << m_dx_backward[0] << "," << m_dx_backward[1] << ","
				<< m_dx_backward[2] << "]" << std::endl;
	}
};

}
}

#endif /*GLOBALDATA_H_*/
