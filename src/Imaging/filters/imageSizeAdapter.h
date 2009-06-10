#ifndef IMAGE_SIZE_ADAPTER_H_
#define IMAGE_SIZE_ADAPTER_H_

#include "common/Common.h"
#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include "Imaging/interpolators/nearestNeighbor.h"

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MedianFilter.h 
 * @{ 
 **/

namespace Imaging
{

/**
 * Shall shrink given image dataset to fit given size.
 * Performs resolution change.
 * NOTE: output elemtents are cast to floats
 */
template< typename ImageType >
class ImageSizeAdapter
	: public AbstractImageFilterWholeAtOnce< ImageType, Image<float32, ImageType::Dimension> >
{
public:	
	typedef ImageSizeAdapter<ImageType> Self;
	typedef float32 TOutPixel;
	typedef Image<float32, ImageType::Dimension> OutImageType;
	//typedef typename ImageType::Element TOutPixel;
	//typedef Image<TOutPixel, ImageType::Dimension> OutImageType;
	typedef AbstractImageFilterWholeAtOnce< ImageType, OutImageType > PredecessorType;
	typedef typename ImageTraits< ImageType >::ElementType ElementType;

	struct Properties : public PredecessorType::Properties
	{
#define DEFAULT_DESIRED_SIZE (256*256*2) // at least 2 slices 256x256
		Properties() : desiredSize( DEFAULT_DESIRED_SIZE ) {}
		Properties(size_t	desiredSize_): desiredSize( desiredSize_) {}
		
		size_t	desiredSize;
	};

	ImageSizeAdapter( Properties * prop );

	GET_SET_PROPERTY_METHOD_MACRO( size_t, DesiredSize, desiredSize );
protected:

	void PrepareOutputDatasets(void);
	bool ProcessImage(const ImageType &in, OutImageType &out);

private:
	
	Vector<float, ImageType::Dimension> _ratio;

	GET_PROPERTIES_DEFINITION_MACRO;
	
	void Process3DImage(Self *self);

};

//include implementation
#include "imageSizeAdapter.tcc"

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*DECIMATION_H_*/
