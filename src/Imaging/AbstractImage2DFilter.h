/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractImage2DFilter.h 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_2D_FILTER_H
#define _ABSTRACT_IMAGE_2D_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImageSliceFilter.h"
#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include <vector>

namespace M4D
{

namespace Imaging
{

// We disallow general usage of template - only specializations.
template< typename InputImageType, typename OutputImageType >
class AbstractImage2DFilter;

/**
 * This template is planned to be used as predecessor for filters procesing on two dimensional data.
 * By that are meant 2D images and 3D images processed in slices. 
 * Output dataset proportions are set to the same values as input dataset, so only method to be overrided
 * is Process2D();
 *
 * This is specialization for 2D images.
 **/
template< typename InputElementType, typename OutputElementType >
class AbstractImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
	 : public AbstractImageFilterWholeAtOnceIExtents< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
{
public:
	typedef AbstractImageFilterWholeAtOnceIExtents< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >	PredecessorType;
	
	struct Properties : public PredecessorType::Properties
	{
		Properties() {}
	};
	
	AbstractImage2DFilter( Properties *prop );
	~AbstractImage2DFilter() {}

protected:
	/**
	 * Computation method for 2D area specified by parameters.
	 * \param inPointer Pointer to element [0,0] in input image.
	 * \param i_xStride Number which should be added to pointer to increase X coordinate in input image.
	 * \param i_yStride Number which should be added to pointer to increase Y coordinate in input image.
	 * \param outPointer Pointer to element [0,0] in output image.
	 * \param o_xStride Number which should be added to pointer to increase X coordinate in output image.
	 * \param o_yStride Number which should be added to pointer to increase Y coordinate in output image.
	 * \param width Width of image area.
	 * \param height Height of image area.
	 * \return Whether computation was succesful.
	 **/
	virtual bool
	Process2D(
			InputElementType	*inPointer,
			int32			i_xStride,
			int32			i_yStride,
			OutputElementType	*outPointer,
			int32			o_xStride,
			int32			o_yStride,
			uint32			width,
			uint32			height
		 ) = 0;

	bool
	ProcessImage(
			const Image< InputElementType, 2 >	&in,
			Image< OutputElementType, 2 >		&out
		    );



private:

};



/**
 * This template is planned to be used as predecessor for filters procesing on two dimensional data.
 * By that are meant 2D images and 3D images processed in slices. 
 * Output dataset proportions are set to the same values as input dataset, so only method to be overrided
 * is Process2D();
 *
 * This is specialization for 3D images.
 **/
template< typename InputElementType, typename OutputElementType >
class AbstractImage2DFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
	 : public AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
{
public:
	typedef AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >	PredecessorType;
	
	struct Properties : public PredecessorType::Properties
	{
		Properties() : PredecessorType::Properties( 0, 10 ) {}
	};
	
	AbstractImage2DFilter( Properties *prop );
	~AbstractImage2DFilter() {}

protected:

	/**
	 * Computation method for 2D area specified by parameters.
	 * \param inPointer Pointer to element [0,0] in input image.
	 * \param i_xStride Number which should be added to pointer to increase X coordinate in input image.
	 * \param i_yStride Number which should be added to pointer to increase Y coordinate in input image.
	 * \param outPointer Pointer to element [0,0] in output image.
	 * \param o_xStride Number which should be added to pointer to increase X coordinate in output image.
	 * \param o_yStride Number which should be added to pointer to increase Y coordinate in output image.
	 * \param width Width of image area.
	 * \param height Height of image area.
	 * \return Whether computation was succesful.
	 **/
	virtual bool
	Process2D(
			InputElementType	*inPointer,
			int32			i_xStride,
			int32			i_yStride,
			OutputElementType	*outPointer,
			int32			o_xStride,
			int32			o_yStride,
			uint32			width,
			uint32			height
		 ) = 0;

	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< OutputElementType, 3 >		&out,
			int32					x1,
			int32					y1,
			int32					x2,
			int32					y2,
			int32					slice
		    );



private:

};


} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/AbstractImage2DFilter.tcc"

#endif /*_ABSTRACT_IMAGE_SLICE_FILTER_H*/

/** @} */

