/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file PipelineContainer.h
 * @{
 **/

#ifndef _PIPELINE_CONTAINER_H
#define _PIPELINE_CONTAINER_H


#include "MedV4D/Imaging/AFilter.h"
#include "MedV4D/Imaging/ConnectionInterface.h"
#include "MedV4D/Imaging/ADataset.h"
#include <memory>
#include <vector>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging {

class PipelineContainer
{
public:
        PipelineContainer();

        virtual
        ~PipelineContainer();

        void
        AddFilter ( APipeFilter *filter );

        void
        AddConnection ( ConnectionInterface *connection );

        void
        ExecuteFirstFilter() {
                _filters[0]->Execute();
        }

        void
        StopFilters();

        void
        Reset();

        /**
         * Connect two compatible ports if possible.
         * @param outPort Reference to output port of some filter.
         * @param inPort Reference to input port of some filter.
         **/
        ConnectionInterface &
        MakeConnection ( M4D::Imaging::OutputPort& outPort, M4D::Imaging::InputPort& inPort );

        ConnectionInterface &
        MakeConnection ( M4D::Imaging::APipeFilter& producer, unsigned producerPortNumber,
                         M4D::Imaging::APipeFilter& consumer, unsigned consumerPortNumber );

        ConnectionInterface &
        MakeInputConnection ( M4D::Imaging::InputPort& inPort, bool ownsDataset );

        ConnectionInterface &
        MakeInputConnection ( M4D::Imaging::APipeFilter& consumer, unsigned consumerPortNumber, bool ownsDataset );

        ConnectionInterface &
        MakeInputConnection ( M4D::Imaging::APipeFilter& consumer, unsigned consumerPortNumber, ADataset::Ptr dataset );

        ConnectionInterface &
        MakeOutputConnection ( M4D::Imaging::OutputPort& outPort, bool ownsDataset );

        ConnectionInterface &
        MakeOutputConnection ( M4D::Imaging::APipeFilter& producer, unsigned producerPortNumber, bool ownsDataset );

        ConnectionInterface &
        MakeOutputConnection ( M4D::Imaging::APipeFilter& producer, unsigned producerPortNumber, ADataset::Ptr dataset );
protected:
        typedef std::vector< APipeFilter * > FilterVector;
        typedef std::vector< ConnectionInterface * > ConnectionVector;

        FilterVector		_filters;
        ConnectionVector	_connections;

private:

};

class EAutoConnectingFailed
{
        //TODO
};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_PIPELINE_H*/


/** @} */

