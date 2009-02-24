#ifndef EXECUTIONDONECALLBACK_H_
#define EXECUTIONDONECALLBACK_H_

#include "server.h"
#include "Imaging/PipelineMessages.h"

namespace M4D
{
namespace RemoteComputing
{

class ExecutionDoneCallback 
  : public M4D::Imaging::MessageReceiverInterface
{
	Server *server_;
public:
	ExecutionDoneCallback( Server *server)
	: server_( server)
		{}

	void ReceiveMessage( 
		M4D::Imaging::PipelineMessage::Ptr    msg, 
		M4D::Imaging::PipelineMessage::MessageSendStyle  sendStyle, 
		M4D::Imaging::FlowDirection    direction
		) 
	{
		switch( msg->msgID)
		{
		case M4D::Imaging::PMI_FILTER_CANCELED:
			server_->OnExecutionFailed();
			break;
		case M4D::Imaging::PMI_FILTER_UPDATED:
			D_PRINT("PMI_FILTER_UPDATED recieved. EXEC COMPLETE !!!");
			server_->OnExecutionDone();
			break;
		default:
			/*ignore other messages*/
      			break;
		}
	}
};

}
}
#endif /*EXECUTIONDONECALLBACK_H_*/
