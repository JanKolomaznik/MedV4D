/**
 * @ingroup imaging 
 * @author Szabolcs Grof
 * @file ImageRegistration.h 
 * @{ 
 **/

#ifndef IMAGE_REGISTRATION_H_
#define IMAGE_REGISTRATION_H_

#include "MedV4D/Imaging/filters/ImageTransform.h"
#include "MedV4D/Imaging/MultiHistogram.h"
#include "MedV4D/Imaging/criterion/NormalizedMutualInformation.h"
#include "MedV4D/Imaging/optimization/PowellOptimization.h"
#include <boost/shared_ptr.hpp>

#define	HISTOGRAM_MIN_VALUE						0
#define HISTOGRAM_MAX_VALUE						200
#define HISTOGRAM_VALUE_DIVISOR						10
#define TRANSFORM_SAMPLING						20

#define MIN_SAMPLING							20

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

/**
 * Class that registers one image to another
 */
template< typename ElementType, uint32 dim >
class ImageRegistration
	: public ImageTransform< ElementType, dim >
{
public:
	typedef Image< ElementType, dim >				ImageType;
	typedef ImageTransform< ElementType, dim >		 	PredecessorType;
	typedef float64							HistCellType;

	/**
	 * Object properties
	 */
	struct Properties : public PredecessorType::Properties
	{
		Properties() {}

	};

	/**
         * Constructor
         *  @param prop pointer to the properties structure
         */
	ImageRegistration( Properties  * prop );

	/**
         * Constructor
         *  @param prop pointer to the properties structure
         */
	ImageRegistration();

	/**
	 * Destructor
	 */
	~ImageRegistration();

	/**
	 * Set the reference image to which the input image from the input port will be registered
	 *  @param ref smart pointer to the reference image
	 */
	void
	SetReferenceImage( AImage::Ptr ref );

	/**
	 * The optimization function that is to be optimized to align the images
	 *  @param v the vector of input parameters of the optimization function
	 *  @return the return value of the optimization function
	 */
	double
	OptimizationFunction( Vector< double, 2 * dim >& v );

	/**
	 * Set the transform sampling while the registration is running
	 *  @param tSampling transform sampling
	 */
	void
	SetTransformSampling( uint32 tSampling );

	/**
	 * Set if automatic registration is requested
	 *  @param mode true if automatic registration is needed, false otherwise
	 */
	void
	SetAutomaticMode( bool mode );

protected:

	/**
	 * The method executed by the pipeline's filter execution thread
	 *  @param  utype update type
         *  @return true if the filter finished successfully, false otherwise
         */
	bool
	ExecutionThreadMethod( APipeFilter::UPDATE_TYPE utype );

	/**
         * Method called in execution methods before actual computation.
         * When overriding in successors predecessor implementation must be called first.
         * \param utype Input/output parameter choosing desired update method. If
         * desired update method can't be used - right type is put as output value.
         **/
        void
        BeforeComputation( APipeFilter::UPDATE_TYPE &utype );

private:
	GET_PROPERTIES_DEFINITION_MACRO;

	/**
	 * Smart pointer to the reference image
	 */
	AImage::Ptr						referenceImage;

	/**
	 * Joint histogram of the two images
	 */
	MultiHistogram< HistCellType, 2 >				jointHistogram;

	/**
	 * Pointer to the criterion component
	 */
	CriterionBase< HistCellType >					*_criterion;

	/**
	 * Pointer to the optimization component
	 */
	OptimizationBase< ImageRegistration< ElementType, dim >, double, 2 * dim >		*_optimization;

	/**
	 * Automatic mode request indicator
	 */
	bool								_automatic;

	/**
	 * Transform sampling while the registration is calculated
	 */
	uint32								_transformSampling;

};

	
} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "MedV4D/Imaging/filters/ImageRegistration.tcc"

#endif /*IMAGE_REGISTRATION_H_*/

/** @} */

