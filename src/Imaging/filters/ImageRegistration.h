/**
 * @ingroup imaging 
 * @author Szabolcs Grof
 * @file ImageRegistration.h 
 * @{ 
 **/

#ifndef IMAGE_REGISTRATION_H_
#define IMAGE_REGISTRATION_H_

#include "Imaging/filters/ImageTransform.h"
#include "Imaging/MultiHistogram.h"
#include "Imaging/criterion/NormalizedMutualInformation.h"
#include "Imaging/optimization/PowellOptimization.h"
#include <boost/shared_ptr.hpp>

#define	HISTOGRAM_MIN_VALUE						0
#define HISTOGRAM_MAX_VALUE						200
#define HISTOGRAM_VALUE_DIVISOR						10
#define HISTOGRAM_SAMPLING_DIVISOR					20

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< typename ElementType, uint32 dim >
class ImageRegistration
	: public ImageTransform< ElementType, dim >
{
public:
	typedef Image< ElementType, dim >				ImageType;
	typedef ImageTransform< ElementType, dim >		 	PredecessorType;
	typedef float64							HistCellType;

	struct Properties : public PredecessorType::Properties
	{
		Properties() {}

	};

	ImageRegistration( Properties  * prop );
	ImageRegistration();

	~ImageRegistration();

	void
	SetReferenceImage( typename ImageType::Ptr ref );

	double
	OptimizationFunction( Vector< double, 2 * dim >& v );

	void
	SetAutomaticMode( bool mode );

protected:
	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );

	/**
         * Method called in execution methods before actual computation.
         * When overriding in successors predecessor implementation must be called first.
         * \param utype Input/output parameter choosing desired update method. If
         * desired update method can't be used - right type is put as output value.
         **/
        void
        BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

private:
	GET_PROPERTIES_DEFINITION_MACRO;

	typename ImageType::Ptr						referenceImage;
	MultiHistogram< HistCellType, 2 >				jointHistogram;
	CriterionBase< HistCellType >					*_criterion;
	OptimizationBase< ElementType, double, 2 * dim >		*_optimization;
	bool								_automatic;

};

	
} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/ImageRegistration.tcc"

#endif /*IMAGE_REGISTRATION_H_*/

/** @} */

