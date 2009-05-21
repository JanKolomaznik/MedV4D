/**
 * Based on the powell algorithm source code, presented in the Numerical Recipes in C++ book.
 */
#ifndef POWELL_OPTIMIZATION_H
#define POWELL_OPTIMIZATION_H

#include <cmath>
#include <limits>
#include "nr/nrutil.h"
#include "common/Common.h"
#include "common/Vector.h"
#include "OptimizationBase.h"

namespace M4D
{
namespace Imaging
{

template< typename RegistrationFilterElementType, typename ElementType, uint32 dim >
class PowellOptimization : public OptimizationBase< RegistrationFilterElementType, ElementType, dim >
{

public:

	typedef 	ElementType (M4D::Imaging::PowellOptimization< RegistrationFilterElementType, ElementType, dim >::*VectorFunc)(NRVec< ElementType > &);
	typedef		ElementType (M4D::Imaging::PowellOptimization< RegistrationFilterElementType, ElementType, dim >::*SingleFunc)(const ElementType);

	void optimize(Vector< ElementType, dim > &v, ElementType &fret, ImageRegistration< RegistrationFilterElementType, dim/2 >* filt );

private:
	void powell(NRVec< ElementType > &p, NRMat< ElementType > &xi, const ElementType ftol, int &iter,
        	ElementType &fret, VectorFunc func);

	void linmin(NRVec< ElementType > &p, NRVec< ElementType > &xi, ElementType &fret, VectorFunc func);

	void mnbrak(ElementType &ax, ElementType &bx, ElementType &cx, ElementType &fa, ElementType &fb, ElementType &fc,
        	SingleFunc func);

	ElementType f1dim(const ElementType x);

	ElementType brent(const ElementType ax, const ElementType bx, const ElementType cx, SingleFunc f,
        	const ElementType tol, ElementType &xmin);

	ElementType caller(NRVec< ElementType > & vec)
	{
		Vector< ElementType, dim > v;
		for ( uint32 i = 0; i < dim; ++i ) v[i] = vec[i];
		return _filter->OptimizationFunction( v );
	}



	inline void shft3(ElementType &a, ElementType &b, ElementType &c, const ElementType d)
        {
                a=b;
                b=c;
                c=d;
        }

	int ncom;
	VectorFunc nrfunc;
	NRVec< ElementType > *pcom_p,*xicom_p;
	ImageRegistration< RegistrationFilterElementType, dim/2 >* _filter;
};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "src/PowellOptimization.tcc"

#endif /*POWELL_OPTIMIZATION_H*/