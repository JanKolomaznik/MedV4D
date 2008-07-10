#ifndef _PIPELINE_MESSAGES_H
#define _PIPELINE_MESSAGES_H

namespace M4D
{
namespace Imaging
{

enum FlowDirection{ FD_IN_FLOW, FD_AGAINST_FLOW };

enum PipelineMsgID
{
	PMI_FILTER_UPDATED,
	PMI_FILTER_START_MODIFICATION,
	PMI_FILTER_CANCELED,

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
	virtual
	~MessageReceiverInterface(){}
	/**
	 * Method for receiving messages - called by sender.
	 * \param msg Smart pointer to message object - we don't have 
	 * to worry about deallocation.
	 * \param sendStyle How treat incoming message.
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


#endif /*_PIPELINE_MESSAGES_H*/
