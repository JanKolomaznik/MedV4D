/**
 * @ingroup imaging 
 * @author Szabolcs Grof
 * @file ImageTransform.h 
 * @{ 
 **/

#ifndef IMAGE_TRANSFORM_H_
#define IMAGE_TRANSFORM_H_

#include "common/Common.h"
#include "Imaging/AbstractImageFilter.h"
#include "Imaging/interpolators/base.h"
#include "Imaging/interpolators/linear.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

/**
 * Filter that transforms the input dataset
 */
template< typename ElementType, uint32 dim >
class ImageTransform
	: public AbstractImageFilter< AbstractImage, Image< ElementType, dim > >
{
public:
	typedef Image< ElementType, dim >			ImageType;
	typedef AbstractImageFilter< AbstractImage, ImageType > 	PredecessorType;
	typedef typename InterpolatorBase< ImageType >::CoordType	CoordType;

	class EDatasetTransformImpossible
	{
		//TODO
	};

	/**
         * Object properties
         */
	struct Properties : public PredecessorType::Properties
	{

		/**
                 * Rotation
                 */
		CoordType _rotation;

		/**
		 * Sampling
		 */
		CoordType _sampling;

		/**
		 * Scaling
		 */
		CoordType _scale;

		/**
		 * Translation
		 */
		CoordType _translation;

		/**
		 * Constructor - fills up the properties with default values
		 */
		Properties()
		{
			for ( uint32 i = 0; i < dim; i++ )
			{
				_rotation[i] = 0.0f;
				_sampling[i] = 1.0f;
				_scale[i] = 1.0f;
				_translation[i] = 0.0f;
			}
		}

	};

	/**
	 * Constructor
	 *  @param prop pointer to the properties structure
	 */
	ImageTransform( Properties  * prop );

	/**
	 * Constructor
	 */
	ImageTransform();

	/**
	 * Set the number of threads performing slice transformations at once
	 *  @param tNumber the number of threads to run simultaneously
	 */
	void SetThreadNumber( uint32 tNumber );

	/**
	 * Set the interpolator to be used during transformation
	 *  @param interpolator pointer to the interpolator to be used
	 */
	void SetInterpolator( InterpolatorBase< ImageType >* interpolator );

	/**
	 * Set the rotation
	 *  @param rotation the rotation to be set
	 */
	void SetRotation(CoordType rotation)
	{
		dynamic_cast< Properties* >( this->_properties )->_rotation = rotation;
	}

	/**
	 * Set the sampling 
	 *  @param sampling the sampling to be set
	 */
	void SetSampling(CoordType sampling)
	{
		dynamic_cast< Properties* >( this->_properties )->_sampling = sampling;
	}

	/**
	 * Set the scale
	 *  @param scale the scale to be set
	 */
	void SetScale(CoordType scale)
	{
		dynamic_cast< Properties* >( this->_properties )->_scale = scale;
	}

	/**
	 * Set the translation
	 *  @param translation the translation to be set
	 */
	void SetTranslation(CoordType translation)
	{
		dynamic_cast< Properties* >( this->_properties )->_translation = translation;
	}

protected:

	/**
	 * Execute the transformation
	 *  @param transformSampling how many voxels should be left out in a row and in a column after
	 *         one interation of voxel transformation
	 *  @return true if the transformation was successful, false otherwise
	 */
	bool
	ExecuteTransformation( uint32 transformSampling = 0 );

	/**
	 * Rescale the output image according to the scaling parameter
	 */
	void
	Rescale();

	/**
	 * This method is executed by the pipeline's filter thread
	 *  @param utype update type
	 *  @return true if the filter finished successfully, false otherwise
	 */
	virtual bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );

	/**
	 * Method which will prepare datasets connected by output ports.
	 * Set their extents, etc.
	 **/
	void
	PrepareOutputDatasets();

	/**
	 * Method called in execution methods before actual computation.
	 * When overriding in successors predecessor implementation must be called first.
	 * \param utype Input/output parameter choosing desired update method. If 
	 * desired update method can't be used - right type is put as output value.
	 **/
	virtual void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	void
	MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype );

	/**
	 * Method called in execution methods after computation.
	 * When overriding in successors, predecessor implementation must be called as last.
	 * \param successful Information, whether computation proceeded without problems.
	 **/
	void
	AfterComputation( bool successful );

	ReaderBBoxInterface::Ptr	_readerBBox;
	WriterBBoxInterface		*_writerBBox;

	/**
	 * The number of threads running simultaneously transforming slices
	 */
	uint32				_threadNumber;

	/**
	 * The interpolator used for transformation
	 */
	InterpolatorBase< ImageType >*	_interpolator;

private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

	
} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/ImageTransform.tcc"

#endif /*IMAGE_TRANSFORM_H_*/

/** @} */

