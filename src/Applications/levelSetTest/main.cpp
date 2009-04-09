#include "common/Common.h"
#include <fstream>

#include "Imaging/AbstractFilter.h"
#include "Imaging/DataSetFactory.h"
#include "Imaging/PipelineContainer.h"
#include "common/IOStreams.h"
#include "remoteComp/remoteServerFilters/levelsetSegmentation/medevedWrapperFilter.h"

using namespace M4D::Common;
using namespace M4D::RemoteComputing;
using namespace M4D::Imaging;
using namespace M4D::IO; 

#define OUTFILE "out.dat"
#define INFILE "in.dat"

class ExecutionDoneCallback 
  : public M4D::Imaging::MessageReceiverInterface
{
	const AbstractDataSet &m_ds;
public:
	ExecutionDoneCallback(const AbstractDataSet &ds)
	: m_ds( ds)
		{}
	
	void SaveDS()
	{
		FOutStream stream(OUTFILE);
		DataSetFactory::SerializeDataset(stream, m_ds);
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
			D_PRINT("PMI_FILTER_UPDATED recieved. EXEC FAILED!!!");
			break;
		case M4D::Imaging::PMI_FILTER_UPDATED:
			D_PRINT("PMI_FILTER_UPDATED recieved. EXEC COMPLETE !!!");
			SaveDS();
			break;
		default:
			/*ignore other messages*/
      			break;
		}
	}
};


int main(int argc, char *argv[]) {
	
	std::ofstream logFile("Log.txt");
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );
	
	M4D::Imaging::AbstractPipeFilter *m_filter;
    M4D::Imaging::ConnectionInterface *m_connWithOutputDataSet;
    M4D::Imaging::PipelineContainer m_pipeLine;    

	try {
		
		M4D::IO::FInStream inStr(INFILE);		
		AbstractDataSet::Ptr inputDataSet = DataSetFactory::DeserializeDataset(inStr);
		
		LevelSetRemoteProperties<int16, int16> *props = new LevelSetRemoteProperties<int16, int16>();
		m_filter =  new ThreshLSSegMedvedWrapper< int16, int16>( props);

		D_PRINT("Connecting recieved dataset into pipeline");
		// connect it to pipeline
		m_pipeLine.MakeInputConnection( *m_filter, 0, inputDataSet);

		// create and connect created output dataSet
		D_PRINT("Creating output connection")
		m_connWithOutputDataSet = &m_pipeLine.MakeOutputConnection( *m_filter, 0,
				true);

		// add message listener to be able catch execution done or failed messages
		MessageReceiverInterface::Ptr messasgeReciever = MessageReceiverInterface::Ptr(
				new ExecutionDoneCallback(m_connWithOutputDataSet->GetDataset()));
		m_connWithOutputDataSet->SetMessageHook( messasgeReciever);
		
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
