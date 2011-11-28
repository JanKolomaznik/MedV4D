#ifndef _ABSTRACT_MULTI_IMAGE_FILTER_H
#define _ABSTRACT_MULTI_IMAGE_FILTER_H

#include "MedV4D/Common/Common.h"

#include "MedV4D/Imaging/Ports.h"
#include "MedV4D/Imaging/ImageDataTemplate.h"
#include "MedV4D/Imaging/ImageFactory.h"
#include "MedV4D/Imaging/AFilter.h"
#include "MedV4D/Imaging/ModificationManager.h"
#include "MedV4D/Imaging/ImageTraits.h"

#include <vector>

namespace M4D
{

/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file AMultiImageFilter.h
 * @{
 **/

namespace Imaging {

template< uint32 InCount, uint32 OutCount >
class AMultiImageFilter: public APipeFilter
{
public:
        typedef APipeFilter	PredecessorType;

        struct Properties : public PredecessorType::Properties {

        };

        ~AMultiImageFilter() {}
protected:
        AMultiImageFilter ( Properties * prop );

        const AImage&
        GetInputImage ( uint32 idx ) const;

        void
        ReleaseInputImage ( uint32 idx ) const;

        AImage&
        GetOutputImage ( uint32 idx ) const;

        void
        ReleaseOutputImage ( uint32 idx ) const;

        void
        SetOutputImageSize (
                uint32		idx,
                uint32		dim,
                int32 		minimums[],
                int32 		maximums[],
                float32		elementExtents[]
        );

        void
        BeforeComputation ( APipeFilter::UPDATE_TYPE &utype );

        void
        AfterComputation ( bool successful );


        const AImage	*in[ InCount ];
        Common::TimeStamp	_inTimestamp[ InCount ];
        Common::TimeStamp	_inEditTimestamp[ InCount ];


        AImage		*out[ OutCount ];
        Common::TimeStamp	_outTimestamp[ OutCount ];
        Common::TimeStamp	_outEditTimestamp[ OutCount ];


private:
        /**
         * Prohibition of copying.
         **/
        PROHIBIT_COPYING_OF_OBJECT_MACRO ( AMultiImageFilter );
};

} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/

//include implementation
#include "MedV4D/Imaging/AMultiImageFilter.tcc"

#endif /*_ABSTRACT_MULTI_IMAGE_FILTER_H*/

