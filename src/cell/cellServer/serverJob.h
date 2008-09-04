/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file serverJob.h 
 * @{ 
 **/

#ifndef SERVERJOB_H
#define SERVERJOB_H

#include "cellBE/basicJob.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/PipelineMessages.h"

#include <map>

namespace M4D
{
namespace CellBE
{

class JobManager;  // forward

class ServerJob
  : public BasicJob
{
  friend class Server;
  friend class ExecutionDoneCallback;

  typedef std::map<uint16, AbstractFilterSerializer *> FilterSerializersMap;

  FilterSerializersMap m_filterSeralizersMap;
  
private:
  ServerJob( boost::asio::ip::tcp::socket *sock, JobManager* jobManager);

  std::vector<uint8> m_filterSettingContent;

  M4D::Imaging::WriterBBoxInterface *m_DSLock;

  M4D::Imaging::PipelineContainer m_pipeLine;
  
  JobManager *m_jobManager;

  // pointers to first & last filter in pipeline
  M4D::Imaging::AbstractPipeFilter *m_pipelineBegin, *m_pipelineEnd;

  void OnDSRecieved( void);

  void DeserializeFilterProperties( void);
  
  /**
   *  Start async operation for definition vector of filters.
   */
  void ReadFilters( void);

  /**
   *  Start async operation for dataSetProperties reading.
   */
  void ReadDataSet( void);

  /**
   *  Calls StopFilters method of PipelineContainer object
   */
  void AbortComputation( void);

  void Execute( void);

  /**
   *  ReadPipelineDefinition
   */
  void ReadPipelineDefinition( void);

  void EndFiltersRead( const boost::system::error_code& error);
  void EndDataSetPropertiesRead( const boost::system::error_code& error);
  void EndReadPipelineDefinition( const boost::system::error_code& error);
  
  /**
   *  Sends result message back to client. Within the method is switch
   *  according result param. If OK, resulting dataSet is then send.
   *  Else just apropriate ResponseID is sent defining the error with
   *  no other data on the tail. Also resultPropertiesLen item of ResultHeader
   *  is not used.
   */
  void SendResultBack( ResponseID result, State state);
  void OnResultHeaderSent( const boost::system::error_code& error
    , ResponseHeader *h);

  void OnExecutionDone( void);
  void OnExecutionFailed( void);

  void WaitForCommand( void);
  void EndWaitForCommand( const boost::system::error_code& error);
  void Command( PrimaryJobHeader *header);
};

///////////////////////////////////////////////////////////////////////////////

class ExecutionDoneCallback 
  : public M4D::Imaging::MessageReceiverInterface
{
	ServerJob *m_job;
public:
	ExecutionDoneCallback( ServerJob *job)
	: m_job( job)
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
			m_job->OnExecutionFailed();
			break;
		case M4D::Imaging::PMI_FILTER_UPDATED:
			D_PRINT("PMI_FILTER_UPDATED recieved. EXEC COMPLETE !!!");
			m_job->OnExecutionDone();
			break;
		default:
			/*ignore other messages*/
		}
	}
};

} // CellBE namespace
} // M4D namespace

#endif
/** @} */

