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

/**
 * Powell's optimization method
 */
template< typename RegistrationFilterElementType, typename ElementType, uint32 dim >
class PowellOptimization : public OptimizationBase< RegistrationFilterElementType, ElementType, dim >
{

public:

	typedef 	ElementType (M4D::Imaging::PowellOptimization< RegistrationFilterElementType, ElementType, dim >::*VectorFunc)(NRVec< ElementType > &);
	typedef		ElementType (M4D::Imaging::PowellOptimization< RegistrationFilterElementType, ElementType, dim >::*SingleFunc)(const ElementType);

	/**
         * Optimize the given criterion function using powell's method
         *  @param v the input parameters
         *  @param fret the return value
         *  @param fil pointer to the registration filter that has the criterion function to optimize
         */
	void optimize(Vector< ElementType, dim > &v, ElementType &fret, RegistrationFilterElementType* filt );

private:

	/**
	 * Powell's method
	 */
	void powell(NRVec< ElementType > &p, NRMat< ElementType > &xi, const ElementType ftol, int &iter,
        	ElementType &fret, VectorFunc func);

	/**
	 * Linear minimization
	 */
	void linmin(NRVec< ElementType > &p, NRVec< ElementType > &xi, ElementType &fret, VectorFunc func);

	/**
	 * Bracketing the result
	 */
	void mnbrak(ElementType &ax, ElementType &bx, ElementType &cx, ElementType &fa, ElementType &fb, ElementType &fc,
        	SingleFunc func);

	/**
	 * 1D part of the function to optimize
	 */
	ElementType f1dim(const ElementType x);

	/**
	 * Brent's method for optimizing 1D functions
	 */
	ElementType brent(const ElementType ax, const ElementType bx, const ElementType cx, SingleFunc f,
        	const ElementType tol, ElementType &xmin);

	/**
	 * Function that converts external module's vector to MedV4D vector
	 */
	ElementType caller(NRVec< ElementType > & vec)
	{
		Vector< ElementType, dim > v;
		for ( uint32 i = 0; i < dim; ++i ) v[i] = vec[i];
		return _filter->OptimizationFunction( v );
	}


	/**
	 * Shift elements
	 */
	inline void shft3(ElementType &a, ElementType &b, ElementType &c, const ElementType d)
        {
                a=b;
                b=c;
                c=d;
        }

	int ncom;
	VectorFunc nrfunc;
	NRVec< ElementType > *pcom_p,*xicom_p;

	/**
	 * Image registration filter that contains the function to optimize
	 */
	RegistrationFilterElementType* _filter;
};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "src/PowellOptimization.tcc"

#endif /*POWELL_OPTIMIZATION_H*/
