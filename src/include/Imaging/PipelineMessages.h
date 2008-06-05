#ifndef _PIPELINE_MESSAGES_H
#define _PIPELINE_MESSAGES_H

namespace M4D
{
namespace Imaging
{

enum FlowDirection{ FD_IN_FLOW, FD_AGAINST_FLOW };

class PipelineMessage
{
public:
	enum MessageSendStyle{ 
		MSS_NORMAL, 
		MSS_DONTRESEND, 
		MSS_BACKWARDS, 
		MSS_BROADCAST 
	};

	typedef boost::shared_ptr< PipelineMessage > Ptr;

	PipelineMessage();

	virtual
	~PipelineMessage();

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


}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_PIPELINE_MESSAGES_H*/
