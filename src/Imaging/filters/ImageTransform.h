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

#define MULTITHREAD_TRANSFORM			5

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< typename ElementType, uint32 dim >
class ImageTransform
	: public AbstractImageFilter< Image< ElementType, dim >, Image< ElementType, dim > >
{
public:
	typedef Image< ElementType, dim >			ImageType;
	typedef AbstractImageFilter< ImageType, ImageType > 	PredecessorType;
	typedef typename InterpolatorBase< ImageType >::CoordType	CoordType;

	class EDatasetTransformImpossible
	{
		//TODO
	};

	struct Properties : public PredecessorType::Properties
	{
		CoordType _rotation;

		CoordType _sampling;

		CoordType _scale;

		CoordType _translation;

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

	ImageTransform( Properties  * prop );
	ImageTransform();

	void SetRotation(CoordType rotation)
	{
		dynamic_cast< Properties* >( this->_properties )->_rotation = rotation;
	}

	void SetSampling(CoordType sampling)
	{
		dynamic_cast< Properties* >( this->_properties )->_sampling = sampling;
	}

	void SetScale(CoordType scale)
	{
		dynamic_cast< Properties* >( this->_properties )->_scale = scale;
	}

	void SetTranslation(CoordType translation)
	{
		dynamic_cast< Properties* >( this->_properties )->_translation = translation;
	}

protected:
	bool
	ExecuteTransformation( uint32 transformSampling );

	void
	Rescale();

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

