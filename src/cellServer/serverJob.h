#ifndef SERVERJOB_H
#define SERVERJOB_H

#include "cellBE/basicJob.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/PipelineMessages.h"

namespace M4D
{
namespace CellBE
{

class ServerJob
  : public BasicJob
{
  friend class Server;
  friend class ExecutionDoneCallback;
  
private:
  ServerJob(boost::asio::io_service &service);

  std::vector<uint8> m_filterSettingContent;

  M4D::Imaging::PipelineContainer m_pipeLine;

  M4D::Imaging::AbstractPipeFilter *m_pipelineBegin, *m_pipelineEnd;

  void DeserializeFilterPropertiesAndBuildPipeline( void);
  
  void ReadFilters( void);
  void ReadDataSet( void);

  void EndFiltersRead( const boost::system::error_code& error);
  void EndDataSetPropertiesRead( const boost::system::error_code& error);
  
  /**
   *  Sends result message back to client. Within the method is switch
   *  according result param. If OK, resulting dataSet is then send.
   *  Else just apropriate ResponseID is sent defining the error with
   *  no other data on the tail. Also resultPropertiesLen item of ResultHeader
   *  is not used.
   */
  void SendResultBack( ResponseID result);
  void OnResultHeaderSent( const boost::system::error_code& error
    , ResponseHeader *h);

  void OnExecutionDone( void);
  void OnExecutionFailed( void);
};

///////////////////////////////////////////////////////////////////////////////

class ExecutionDoneCallback 
  : public M4D::Imaging::MessageReceiverInterface
{
  ServerJob *m_job;
public:
  ExecutionDoneCallback( ServerJob *job)
    : m_job( job)
  {
  }

  void ReceiveMessage( 
    M4D::Imaging::PipelineMessage::Ptr    msg, 
    M4D::Imaging::PipelineMessage::MessageSendStyle  sendStyle, 
    M4D::Imaging::FlowDirection    direction
    ) 
  {
   switch( msg->msgID)
   {
   case M4D::Imaging::PMI_FILTER_CANCELED:
     m_job->OnExecutionFailed();
     break;
   case M4D::Imaging::PMI_FILTER_UPDATED:
     m_job->OnExecutionDone();
     break;
   }
  }
};

} // CellBE namespace
} // M4D namespace

#endif