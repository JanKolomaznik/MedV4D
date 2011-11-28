/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file AImageSliceFilter.h
 * @{
 **/

#ifndef _A_IMAGE_SLICE_FILTER_H
#define _A_IMAGE_SLICE_FILTER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/AImageFilter.h"
#include <vector>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging {

struct SliceComputationRecord {
        ReaderBBoxInterface::Ptr	inputBBox;
        WriterBBoxInterface		*writerBBox;
        int32				firstSlice;
        int32				lastSlice;
};

/**
 * We disallow general usage of template - only specializations.
 **/
template< typename InputImageType, typename OutputImageType >
class AImageSliceFilter;

template< typename InputElementType, typename OutputImageType >
class AImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
                        : public AImageFilter< Image< InputElementType, 3 >, OutputImageType >
{
public:
        typedef AImageFilter< Image< InputElementType, 3 >, OutputImageType >	PredecessorType;
        typedef Image< InputElementType, 3 >	InputImageType;

        struct Properties : public PredecessorType::Properties {
                Properties ( unsigned sliceNeighbourCount, unsigned computationGrouping )
                                : _sliceComputationNeighbourCount ( sliceNeighbourCount ), _computationGrouping ( computationGrouping ) {}
                /**
                 * How many slices to up and down are needed for computation.
                 * This information is needed when waiting for input update.
                 **/
                unsigned	_sliceComputationNeighbourCount;

                /**
                 * How many slices will be put into one computation sequence.
                 **/
                unsigned	_computationGrouping;
        };

        AImageSliceFilter ( Properties *prop );
        ~AImageSliceFilter() {}

        void
        SetComputationNeighbourhood ( unsigned count ) {
                static_cast<Properties*> ( this->_properties )->_sliceComputationNeighbourCount = count;
        }

        unsigned
        GetComputationNeighbourhood() {
                return static_cast<Properties*> ( this->_properties )->_sliceComputationNeighbourCount;
        }

        void
        SetComputationGrouping ( unsigned count ) {
                if ( count > 0 ) {
                        static_cast<Properties*> ( this->_properties )->_computationGrouping = count;
                } else {
                        _THROW_ ErrorHandling::ExceptionBadParameter< unsigned > ( count );
                }
        }

        unsigned
        GetComputationGrouping() {
                return static_cast<Properties*> ( this->_properties )->_computationGrouping;
        }
protected:

        typedef std::vector< SliceComputationRecord >	ComputationGroupList;

        virtual bool
        ProcessSlice (
                const ImageRegion< InputElementType, 3 >	&inRegion,
                OutputImageType					&out,
                int32						slice
        ) = 0;

        virtual WriterBBoxInterface &
        GetComputationGroupWriterBBox ( SliceComputationRecord & record ) = 0;

        bool
        ExecutionThreadMethod ( APipeFilter::UPDATE_TYPE /*utype*/ );


        void
        MarkChanges ( APipeFilter::UPDATE_TYPE utype );



        ComputationGroupList		_actualComputationGroups;
private:
        GET_PROPERTIES_DEFINITION_MACRO;
        /**
         * Prohibition of copying.
         **/
        PROHIBIT_COPYING_OF_OBJECT_MACRO ( AImageSliceFilter );
};



/**
 * We disallow general usage of template - only specializations.
**/
template< typename InputImageType, typename OutputImageType >
class AImageSliceFilterIExtents;

template< typename InputElementType, typename OutputElementType >
class AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
                        : public AImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
{
public:
        typedef AImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >	PredecessorType;
        typedef Image< InputElementType, 3 >	InputImageType;
        typedef Image< OutputElementType, 3 >	OutputImageType;

        struct Properties : public PredecessorType::Properties {
                Properties ( unsigned sliceNeighbourCount, unsigned computationGrouping )
                                : _sliceComputationNeighbourCount ( sliceNeighbourCount ), _computationGrouping ( computationGrouping ) {}
                /**
                 * How many slices to up and down are needed for computation.
                 * This information is needed when waiting for input update.
                 **/
                unsigned	_sliceComputationNeighbourCount;

                /**
                 * How many slices will be put into one computation sequence.
                 **/
                unsigned	_computationGrouping;
        };

        AImageSliceFilterIExtents ( Properties *prop );
        ~AImageSliceFilterIExtents() {}

        void
        SetComputationNeighbourhood ( unsigned count ) {
                static_cast<Properties*> ( this->_properties )->_sliceComputationNeighbourCount = count;
        }

        unsigned
        GetComputationNeighbourhood() {
                return static_cast<Properties*> ( this->_properties )->_sliceComputationNeighbourCount;
        }

        void
        SetComputationGrouping ( unsigned count ) {
                if ( count > 0 ) {
                        static_cast<Properties*> ( this->_properties )->_computationGrouping = count;
                } else {
                        _THROW_ ErrorHandling::ExceptionBadParameter< unsigned > ( count );
                }
        }

        unsigned
        GetComputationGrouping() {
                return static_cast<Properties*> ( this->_properties )->_computationGrouping;
        }
protected:

        typedef std::vector< SliceComputationRecord >	ComputationGroupList;

        void
        PrepareOutputDatasets();

        virtual bool
        ProcessSlice (
                const ImageRegion< InputElementType, 3 >	&inRegion,
                ImageRegion< OutputElementType, 2 > 		&outRegion,
                int32						slice
        ) = 0;

        WriterBBoxInterface &
        GetComputationGroupWriterBBox ( SliceComputationRecord & record );

        bool
        ExecutionThreadMethod ( APipeFilter::UPDATE_TYPE /*utype*/ );


        void
        MarkChanges ( APipeFilter::UPDATE_TYPE utype );



        ComputationGroupList		_actualComputationGroups;
private:
        GET_PROPERTIES_DEFINITION_MACRO;
        /**
         * Prohibition of copying.
         **/
        PROHIBIT_COPYING_OF_OBJECT_MACRO ( AImageSliceFilterIExtents );
};

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "MedV4D/Imaging/AImageSliceFilter.tcc"

#endif /*_A_IMAGE_SLICE_FILTER_H*/

/** @} */

