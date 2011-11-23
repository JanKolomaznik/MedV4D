/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file PipelineMessages.h 
 * @{ 
 **/

#ifndef _PIPELINE_MESSAGES_H
#define _PIPELINE_MESSAGES_H

#include <boost/shared_ptr.hpp>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

enum FlowDirection{ 
			FD_IN_FLOW 		= 1, 
			FD_AGAINST_FLOW		= 1 << 1, 
			FD_BOTH 		= ( 1 | (1 << 1) ) 
		};

enum PipelineMsgID
{
	PMI_FILTER_UPDATED,
	PMI_FILTER_START_MODIFICATION,
	PMI_FILTER_CANCELED,
	PMI_PORT_PLUGGED,
	PMI_DATASET_PUT,
	PMI_DATASET_REMOVED,
	PMI_PROGRESS_INFO,

	PMI_END_SYSMSG = 1000
};


class PipelineMessage
{
public:
	enum MessageSendStyle{ 
		/**
		 * Message will be used by receiver, if he can handle it, 
		 * otherwise hi will resend it to followers.
		 **/
		MSS_NORMAL, 
		/**
		 * Receiver will not resend message even when he can't handle it.
		 **/
		MSS_DONTRESEND, 
		MSS_BACKWARDS, 
		MSS_BROADCAST 
	};

	typedef boost::shared_ptr< PipelineMessage > Ptr;

	PipelineMessage( PipelineMsgID messageID )
		: senderID( 0 ), msgID( messageID ) {}

	virtual
	~PipelineMessage(){};

	uint64	senderID;

	const PipelineMsgID	msgID;

};

class MsgFilterExecutionCanceled: public PipelineMessage
{
public:
	MsgFilterExecutionCanceled(): PipelineMessage( PMI_FILTER_CANCELED )
		{ /*empty*/ }

	static PipelineMessage::Ptr
	CreateMsg()
	{
		//TODO improve
		return PipelineMessage::Ptr( new MsgFilterExecutionCanceled() );
	}

};

class MsgFilterUpdated: public PipelineMessage
{
public:
	MsgFilterUpdated( bool whole ): PipelineMessage( PMI_FILTER_UPDATED ), _whole( whole ) 
		{ /*empty*/ }

	static PipelineMessage::Ptr
	CreateMsg( bool whole )
	{
		//TODO improve
		return PipelineMessage::Ptr( new MsgFilterUpdated( whole ) );
	}

	bool
	IsUpdatedWhole()const
	{ return _whole; }
protected:
	bool	_whole;
};

class MsgFilterStartModification: public PipelineMessage
{
public:
	MsgFilterStartModification( bool whole ): PipelineMessage( PMI_FILTER_START_MODIFICATION ), _whole( whole ) 
		{ /*empty*/ }

	static PipelineMessage::Ptr
	CreateMsg( bool whole )
	{
		//TODO improve
		return PipelineMessage::Ptr( new MsgFilterStartModification( whole ) );
	}

	bool
	IsUpdatedWhole()const
	{ return _whole; }
protected:
	bool	_whole;
};

class MsgPortPlugged: public PipelineMessage
{
public:
	MsgPortPlugged(): PipelineMessage( PMI_PORT_PLUGGED )
		{ /*empty*/ }

	static PipelineMessage::Ptr
	CreateMsg()
	{
		//TODO improve
		return PipelineMessage::Ptr( new MsgPortPlugged() );
	}

};

class MsgDatasetPut: public PipelineMessage
{
public:
	MsgDatasetPut(): PipelineMessage( PMI_DATASET_PUT )
		{ /*empty*/ }

	static PipelineMessage::Ptr
	CreateMsg()
	{
		//TODO improve
		return PipelineMessage::Ptr( new MsgDatasetPut() );
	}

};

class MsgDatasetRemoved: public PipelineMessage
{
public:
	MsgDatasetRemoved(): PipelineMessage( PMI_DATASET_REMOVED )
		{ /*empty*/ }

	static PipelineMessage::Ptr
	CreateMsg()
	{
		//TODO improve
		return PipelineMessage::Ptr( new MsgDatasetRemoved() );
	}

};


class MsgProgressInfo: public PipelineMessage
{
public:
	MsgProgressInfo( uint32 partID, uint32 partCount )
		: PipelineMessage( PMI_PROGRESS_INFO ), _partID( partID ), _partCount( partCount )
		{ /*empty*/ }

	static PipelineMessage::Ptr
	CreateMsg( uint32 partID, uint32 partCount )
	{
		//TODO improve
		return PipelineMessage::Ptr( new MsgProgressInfo( partID, partCount ) );
	}
protected:
	uint32 _partID; 
	uint32 _partCount;
};

//*****************************************************************************

class MessageSenderInterface
{
public:
	virtual
	~MessageSenderInterface(){}
	/**
	 * Send message out from object.
	 * \param msg Smart pointer to message object - we don't have 
	 * to worry about deallocation.
	 * \param sendStyle How receiver should treat the message.
	 **/
	virtual void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle
		) = 0;

};

class MessageRouterInterface
{
public:
	virtual
	~MessageRouterInterface(){}
	/**
	 * Send message out from object.
	 * \param msg Smart pointer to message object - we don't have 
	 * to worry about deallocation.
	 * \param sendStyle How receiver should treat the message.
	 * \param direction If message goes in flow or against flow (or both) of pipeline.
	 **/
	virtual void
	RouteMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle,
		FlowDirection				direction
		) = 0;

};


class MessageReceiverInterface
{
public:
	typedef boost::shared_ptr< MessageReceiverInterface > Ptr;

	virtual
	~MessageReceiverInterface(){}
	/**
	 * Method for receiving messages - called by sender.
	 * \param msg Smart pointer to message object - we don't have 
	 * to worry about deallocation.
	 * \param sendStyle How treat incoming message.
	 * \param direction If message goes in flow or against flow (or both) of pipeline.
	 **/
	virtual void
	ReceiveMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle, 
		FlowDirection				direction
		) = 0;
};


class MessageOperatorInterface : public MessageSenderInterface, public MessageReceiverInterface
{
public:
	~MessageOperatorInterface(){}

};


template< typename SenderTypePointer >
struct MessageSenderFunctor
{
	MessageSenderFunctor( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle
		) : _msg( msg ), _sendStyle( sendStyle ) {}

	void
	operator()( SenderTypePointer sender )
	{
		sender->SendMessage( _msg, _sendStyle );
	}
protected:
	PipelineMessage::Ptr 			_msg;
	PipelineMessage::MessageSendStyle 	_sendStyle;
};

template< typename ReceiverTypePointer >
struct MessageReceiverFunctor
{
	MessageReceiverFunctor( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle,
		FlowDirection				direction
		) : _msg( msg ), _sendStyle( sendStyle ), _direction( direction ) {}

	void
	operator()( ReceiverTypePointer receiver )
	{
		receiver->ReceiveMessage( _msg, _sendStyle, _direction );
	}
protected:
	PipelineMessage::Ptr 			_msg;
	PipelineMessage::MessageSendStyle 	_sendStyle;
	FlowDirection				_direction;
};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_PIPELINE_MESSAGES_H*/

/** @} */

