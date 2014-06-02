/**
 *  @defgroup imaging Imaging Library
 */

/**
 * @ingroup imaging
 * @file AFilter.h
 * @author Jan Kolomaznik
 * @{
 **/

#ifndef _ABSTRACT_FILTER_H
#define _ABSTRACT_FILTER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/AProcessingUnit.h"
#include "MedV4D/Imaging/Ports.h"
#include "MedV4D/Imaging/ConnectionInterface.h"
#include <memory>
#include "MedV4D/Common/Thread.h"
#include "MedV4D/Imaging/PipelineMessages.h"
#include "filterIDsEnum.h"
#include <string>

#include <iostream>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  Main goal of this library is effective implementation of pipeline
 *  computation on input datasets. Whole computation should be as parallel
 *  as possible in order to utilize resources available in modern processors,
 *  etc.
 */

namespace M4D
{
namespace Imaging {
/**
 * Macro used for easy creation of get method to access Properties structure with right type
 **/
#define GET_PROPERTIES_DEFINITION_MACRO \
	Properties & GetProperties(){ return *(static_cast<Properties*>( this->_properties ) ); }

/**
 * Macro unwinding to get method for property.
 * \param TYPE Type of property - return value of the method.
 * \param NAME Name of property used in name of function - Get'NAME'().
 * \param \PROPERTY_NAME Name of property in Properties structure.
 **/
#define GET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME ) \
	TYPE Get##NAME ()const{ return (static_cast<Properties*>( this->_properties ) )->PROPERTY_NAME ; }

/**
 * Macro unwinding to set method for property.
 * \param TYPE Type of property - parameter type of the method.
 * \param NAME Name of property used in name of function - Set'NAME'().
 * \param \PROPERTY_NAME Name of property in Properties structure.
 **/
#define SET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME ) \
	void Set##NAME ( TYPE value ){ this->_properties->IncTimestamp(); (static_cast<Properties*>( this->_properties ) )->PROPERTY_NAME = value; }

/**
 * Macro unwinding to previously defined macros.
 **/
#define GET_SET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME ) \
	GET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME ) \
	SET_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME )
/**
 * Structure synchronizing access to filter state informations.
 **/
struct FilterWorkingState {
public:
        /**
         * Construct state object in UP_TO_DATE state.
         **/
        FilterWorkingState() :_state ( UP_TO_DATE ) {}

        /**
         * Try set state to RUNNING - only possible when in state
         * UP_TO_DATE or OUT_OF_DATE.
         * @return Whether set was successful.
         **/
        bool
        TrySetRunning();

        /**
         * Try set state to STOPPING - only possible when in state
         * RUNNING or already STOPPING.
         * @return Whether set was successful.
         **/
        bool
        TrySetStopping();

        /**
         * Try set state to UP_TO_DATE - only possible when in state
         * OUT_OF_DATE or already UP_TO_DATE.
         * @return Whether set was successful.
         **/
        bool
        TrySetUpToDate();

        /**
         * Try set state to OUT_OF_DATE - only possible when in state
         * UP_TO_DATE or already OUT_OF_DATE.
         * @return Whether set was successful.
         **/
        bool
        TrySetOutOfDate();

        /**
         * Change state unconditionaly.
         **/
        void
        SetRunning();

        /**
         * Change state unconditionaly.
         **/
        void
        SetStopping();

        /**
         * Change state unconditionaly.
         **/
        void
        SetUpToDate();

        /**
         * Change state unconditionaly.
         **/
        void
        SetOutOfDate();

        /**
         * @return True if is in state RUNNING.
         **/
        bool
        IsRunning() const {
                return _state == RUNNING;
        }

protected:

private:
        /**
         * Enumeration of all possible states.
         **/
        enum FILTER_STATE { RUNNING, STOPPING, UP_TO_DATE, OUT_OF_DATE };

        /**
         * Actual state.
         **/
        volatile FILTER_STATE		_state;

        /**
         * Mutex for locking when changing state.
         **/
        Multithreading::Mutex	_stateLock;
};


/**
 * Ancestor of all filters. This class declare basic execution interface for all filters.
 * Its purpose is just ensure common predecessor of pipeline filters and filters with
 * different computation logic.
 **/
class AFilter
                        : public AProcessingUnit
{
public:
        struct Properties {
        public:
                virtual
                ~Properties() {}
                virtual void
                CheckProperties() {}

                void
                IncTimestamp() {
                        ++_timestamp;
                }

                M4D::Common::TimeStamp
                GetTimestamp() const {
                        return _timestamp;
                }
                /**
                 * returns ID of the filter used in filter serialization
                 */
                virtual FilterID GetID ( void ) {
                        return FID_AFilterNOT_USE;
                }

        private:
                M4D::Common::TimeStamp	_timestamp;
        };

        /**
         * Smart pointer to filter with this interface.
         **/
        typedef std::shared_ptr< AFilter > AFilterPtr;
        typedef std::shared_ptr< AFilter > Ptr;

        /**
         * Destructor - virtual - can be polymorphically destroyed.
         **/
        ~AFilter() {
                delete _properties;
        }

        /**
         * Start computing only on modified data.
         * Asynchronous method.
         **/
        virtual void
        Execute() = 0;

        /**
         * Start computing from scratch - recalculate output
         * using all input data, even when no change was applied.
         * Asynchronous method.
         **/
        virtual void
        ExecuteOnWhole() = 0;

        /**
         * Stop execution of filter as soon as possible.
         * Asynchronous method.
         * \return True if stopping call was successful - it means that
         * filter will stop its execution. Otherwise some problem occured -
         * filter couldn't be stopped.
         **/
        virtual bool
        StopExecution() = 0;

        std::string
        GetName() const {
                return _name;
        }
protected:

        AFilter ( AFilter::Properties * prop ) : _properties ( prop ), _name ( "Filter" ) {}

        Properties *_properties;

        std::string _name;

private:
        /**
         * Prohibition of copying.
         **/
        PROHIBIT_COPYING_OF_OBJECT_MACRO ( AFilter );
};

/**
 * Ancestor of pipeline filters. Public interface is extended with few methods modifiing
 * behaviour (ie. setting invocation style) and access methods to input ports and output ports.
 * These ports are comunication channels - can send and receive messages, get access to datasets, etc.
 *
 * In nonpublic interface there are declared pure virtual and virtual methods with special purpose - they are
 * called in predefined situations or in right order during computation.
 * If somebody wants to create new pipeline filter, he must at least inherit its implementation from this class and
 * override these methods : ExecutionThreadMethod(), PrepareOutputDatasets(), BeforeComputation(), MarkChanges(),
 * AfterComputation().
 **/
class APipeFilter : public AFilter, public MessageReceiverInterface
{
public:
        typedef AFilter	PredecessorType;

        typedef PredecessorType::Properties Properties;

        enum UpdateInvocationStyle {
                UIS_ON_DEMAND,
                UIS_ON_CHANGE_BEGIN,
                UIS_ON_UPDATE_FINISHED
        };

        /**
         * Types of possible execution methods - used by execution thread functor.
         **/
        enum UPDATE_TYPE { RECALCULATION, ADAPTIVE_CALCULATION };
        /**
         * Smart pointer to filter with this interface.
         **/
        typedef std::shared_ptr< APipeFilter > APipeFilterPtr;


        /**
         * Destructor - virtual - can be polymorphically destroyed.
         **/
        ~APipeFilter() {}

        /**
         * @return Returns list of all available input ports.
         **/
        const InputPortList &
        InputPort() const {
                return _inputPorts;
        }

        /**
         * @return Returns list of all available output ports.
         **/
        const OutputPortList &
        OutputPort() const {
                return _outputPorts;
        }

        template< typename DatasetType >
        const DatasetType&
        GetInputDataset ( uint32 idx ) const;

        void
        ReleaseInputDataset ( uint32 idx ) const;

        template< typename DatasetType >
        DatasetType &
        GetOutputDataset ( uint32 idx ) const;

        void
        ReleaseOutputDataset ( uint32 idx ) const;

        /**
         * Start computing only on modified data.
         * Asynchronous method.
         **/
        void
        Execute();

        /**
         * Start computing from scretch - recalculate output
         * using all input data, even when no change was applied.
         * Asynchronous method.
         **/
        virtual void
        ExecuteOnWhole();

        /**
         * Stop execution of filter as soon as possible.
         * Asynchronous method.
         * \return True if stopping call was successful - it means that
         * filter will stop its execution. Otherwise some problem occured -
         * filter couldn't be stopped.
         **/
        bool
        StopExecution();

        bool
        IsRunning() const {
                return _workState.IsRunning();
        }

        /**
         * @return True if all computations are done or don't have input data.
         **/
        bool
        IsUpToDate();

        unsigned
        SetParallelization ( unsigned threadCount );

        unsigned
        GetParallelization() const;

        /**
         * Sets ivocation style of filter - when it will be executed.
         * \param style Style to be set.
         **/
        void
        SetUpdateInvocationStyle ( UpdateInvocationStyle style ) {
                _invocationStyle = style;
        }


        /**
         * Method for receiving messages - called by sender.
         * \param msg Smart pointer to message object - we don't have
         * to worry about deallocation.
         * \param sendStyle How treat incoming message.
         * \param direction In what direction was this message sent -
         * in flow or against flow of a pipeline.
         **/
        void
        ReceiveMessage (
                PipelineMessage::Ptr 			msg,
                PipelineMessage::MessageSendStyle 	sendStyle,
                FlowDirection				direction
        );

        SIMPLE_GET_METHOD ( float32, LastComputationTime, _lastComputationTime );

        Properties *GetPropertiesPointer() const {
                return _properties;
        }
protected:
        friend struct MainExecutionThread;


        APipeFilter ( APipeFilter::Properties *prop );

        /**
         * Method running in execution thread - this method will be
         * stopped, when StopExecution() is invoked.
         * In inherited class reimplementation of this method is easy way to
         * implement new filter, and let all dirty work to ancestor class.
         * \param utype Tells how filter computation should proceed - on whole dataset,
         * or update only on changed parts.
         * \return True if execution wasn't stopped, false otherwise.
         **/
        virtual bool
        ExecutionThreadMethod ( APipeFilter::UPDATE_TYPE utype ) =0;

        /**
         * Method used for checking whether execution can continue.
         * Execution threads should call this method often to ensure that
         * StopExecution() method wasn't called.
         * \return True if computing can continue, false otherwise.
         **/
        bool
        CanContinue();

        /**
         * Method used for clean after successful run of execution thread.
         * It will first detach thread and than set _workState to UP_TO_DATE.
         * Because execution thread is detached, new execution thread can be
         * started. The old one will only finish - after this call it doesn't
         * change internal state of filter.
         **/
        void
        CleanAfterSuccessfulRun();

        /**
         * Method used for cleaning after stopped run of execution thread.
         * It will first detach thread and than set _workState to OUT_OF_DATE.
         * Because execution thread is detached, new execution thread can be
         * started. The old one will only finish - after this call it doesn't
         * change internal state of filter.
         **/
        void
        CleanAfterStoppedRun();

        /**
         * Method which will prepare datasets connected by output ports.
         * Set their extents, etc.
         **/
        virtual void
        PrepareOutputDatasets() { /*empty*/ };

        /**
         * Method called in execution methods before actual computation.
         * When overriding in successors predecessor implementation must be called first.
         * \param utype Input/output parameter choosing desired update method. If
         * desired update method can't be used - right type is put as output value.
         **/
        virtual void
        BeforeComputation ( APipeFilter::UPDATE_TYPE &utype );

        virtual void
        MarkChanges ( APipeFilter::UPDATE_TYPE utype ) = 0;

        /**
         * Method called in execution methods after computation.
         * When overriding in successors, predecessor implementation must be called as last.
         * \param successful Information, whether computation proceeded without problems.
         **/
        virtual void
        AfterComputation ( bool successful ) {
                successful = successful;
                _propertiesTimestamp = _properties->GetTimestamp();
        }

        bool
        PropertiesUpdated() const {
                return _propertiesTimestamp != _properties->GetTimestamp();
        }

        void
        InputDatasetUpdatedMsgHandler ( MsgFilterUpdated *msg );

        void
        InputDatasetStartModificationMsgHandler ( MsgFilterStartModification *msg );

        void
        InputDatasetComputationCanceledMsgHandler ( MsgFilterExecutionCanceled *msg );

        /**
         * Container for input ports - polymorphic interfaces.
         **/
        InputPortList		_inputPorts;

        /**
         * Container for output ports - polymorphic interfaces.
         **/
        OutputPortList		_outputPorts;

        /**
         * Main filter thread - computation is executed in this thread,
         * or its child threads.
         **/
        Multithreading::Thread	*_executionThread;

        /**
         * Internal state of filter.
         **/
        FilterWorkingState	_workState;

        /**
         * Invocation style of filter - see comments for possible values.
         **/
        UpdateInvocationStyle 	_invocationStyle;

        M4D::Common::TimeStamp	_propertiesTimestamp;

        bool			_callPrepareOutputDatasets;


        float32			_lastComputationTime;
private:
        /**
         * Prohibition of copying.
         **/
        PROHIBIT_COPYING_OF_OBJECT_MACRO ( APipeFilter );

};

typedef APipeFilter APipeFilter;

template< typename DatasetType >
const DatasetType&
APipeFilter::GetInputDataset ( uint32 idx ) const
{
        _inputPorts[ idx ].LockDataset();
        try {
                const DatasetType* dataset = dynamic_cast< const DatasetType* > ( & ( _inputPorts.GetPort ( idx ).GetDataset() ) );
                if ( dataset ) return *dataset;
        } catch ( ... ) {
                _inputPorts[ idx ].ReleaseDatasetLock();
                throw;
        }

        _inputPorts[ idx ].ReleaseDatasetLock();
        _THROW_ ErrorHandling::ECastProblem();
}

template< typename DatasetType >
DatasetType &
APipeFilter::GetOutputDataset ( uint32 idx ) const
{
        _outputPorts[ idx ].LockDataset();
        try {
                DatasetType* dataset = dynamic_cast< DatasetType* > ( & ( _outputPorts.GetPort ( idx ).GetDataset() ) );
                if ( dataset ) return *dataset;
        } catch ( ... ) {
                _outputPorts[ idx ].ReleaseDatasetLock();
                throw;
        }

        _outputPorts[ idx ].ReleaseDatasetLock();
        _THROW_ ErrorHandling::ECastProblem();
}

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_FILTER_H*/

/** @} */

