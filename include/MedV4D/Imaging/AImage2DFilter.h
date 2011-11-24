/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file AImage2DFilter.h
 * @{
 **/

#ifndef _ABSTRACT_IMAGE_2D_FILTER_H
#define _ABSTRACT_IMAGE_2D_FILTER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/AImageSliceFilter.h"
#include "MedV4D/Imaging/AImageFilterWholeAtOnce.h"
#include <vector>

namespace M4D
{

namespace Imaging {

// We disallow general usage of template - only specializations.
template< typename InputImageType, typename OutputImageType >
class AImage2DFilter;

/**
 * This template is planned to be used as predecessor for filters procesing on two dimensional data.
 * By that are meant 2D images and 3D images processed in slices.
 * Output dataset proportions are set to the same values as input dataset, so only method to be overrided
 * is Process2D();
 *
 * This is specialization for 2D images.
 **/
template< typename InputElementType, typename OutputElementType >
class AImage2DFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
                        : public AImageFilterWholeAtOnceIExtents< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
{
public:
        typedef AImageFilterWholeAtOnceIExtents< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >	PredecessorType;

        struct Properties : public PredecessorType::Properties {
                Properties() {}
        };

        AImage2DFilter ( Properties *prop );
        ~AImage2DFilter() {}

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
        Process2D (
                const ImageRegion< InputElementType, 2 > &inRegion,
                ImageRegion< OutputElementType, 2 > 	 &outRegion
        ) = 0;

        bool
        ProcessImage (
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
class AImage2DFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
                        : public AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
{
public:
        typedef AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >	PredecessorType;

        struct Properties : public PredecessorType::Properties {
                Properties() : PredecessorType::Properties ( 0, 10 ) {}
        };

        AImage2DFilter ( Properties *prop );
        ~AImage2DFilter() {}

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
        Process2D (
                const ImageRegion< InputElementType, 2 >	&inRegion,
                ImageRegion< OutputElementType, 2 >		&outRegion
        ) = 0;

        bool
        ProcessSlice (
                const ImageRegion< InputElementType, 3 >	&inRegion,
                ImageRegion< OutputElementType, 2 > 		&outRegion,
                int32						slice
        );



private:

};


} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "MedV4D/Imaging/AImage2DFilter.tcc"

#endif /*_ABSTRACT_IMAGE_SLICE_FILTER_H*/

/** @} */

