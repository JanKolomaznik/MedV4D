#include "common/Common.h"
#include <fstream>

#include "Imaging/AbstractFilter.h"
#include "Imaging/DataSetFactory.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/filters/ImageConvertor.h"
#include "common/IOStreams.h"
#include "remoteComp/remoteServerFilters/levelsetSegmentation/medevedWrapperFilter.h"

using namespace M4D::Common;
using namespace M4D::RemoteComputing;
using namespace M4D::Imaging;
using namespace M4D::IO;
using namespace M4D::Multithreading;

#define OUTFILE "out.mv4d"
#define INFILE "in.mv4d"

class ExecutionDoneCallback 
  : public M4D::Imaging::MessageReceiverInterface
{
	const AbstractDataSet &m_ds;
public:
	ExecutionDoneCallback(const AbstractDataSet &ds)
	: m_ds( ds)
		{
			lock = new ScopedLock(mutex);
		}
	
	void SaveDS()
	{
		FOutStream stream(OUTFILE);
		DataSetFactory::SerializeDataset(stream, m_ds);
	}
	
	Mutex mutex;	

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
			delete lock;
			break;
		case M4D::Imaging::PMI_FILTER_UPDATED:
			D_PRINT("PMI_FILTER_UPDATED recieved. EXEC COMPLETE !!!");
			SaveDS();
			delete lock;
			break;
		default:
			/*ignore other messages*/
      			break;
		}
	}
	
private:
	ScopedLock *lock;
};


int main(int argc, char *argv[]) {
	
	std::ofstream logFile("Log.txt");
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );
	
	M4D::Imaging::AbstractPipeFilter *m_filter;
//	M4D::Imaging::AbstractPipeFilter *_convertor;
    M4D::Imaging::ConnectionInterface *m_connWithOutputDataSet;
    M4D::Imaging::PipelineContainer m_pipeLine;    

	try {
		AbstractDataSet::Ptr inputDataSet;
		
		{
			M4D::IO::FInStream inStr(INFILE);		
			inputDataSet = DataSetFactory::DeserializeDataset(inStr);
		}
		
		typedef ThreshLSSegMedvedWrapper< float32, float32 > FilterType;
		
		FilterType::Properties *props = new FilterType::Properties();
		
		Image<float32, 3> *im = (Image<float32, 3> *) inputDataSet.get();
		
		props->seedX = 
			(im->GetDimensionExtents( 0 ).maximum - im->GetDimensionExtents( 0 ).minimum) / 2;
		props->seedY = 
			(im->GetDimensionExtents( 1 ).maximum - im->GetDimensionExtents( 1 ).minimum) / 2;		
		props->seedZ = 
			(im->GetDimensionExtents( 2 ).maximum - im->GetDimensionExtents( 2 ).minimum) / 2;
		
		
		
#if( defined(SPU_TIMING_TOOL_PROFILING) )
		props->maxIterations = 1;
		props->initialDistance = 5;
#else
		props->maxIterations = 800;
		props->initialDistance = (double)
					(im->GetDimensionExtents( 0 ).maximum - im->GetDimensionExtents( 0 ).minimum) / 2.2;
#endif
		m_filter =  new FilterType( props);
		
		m_filter->SetUpdateInvocationStyle(AbstractPipeFilter::UIS_ON_CHANGE_BEGIN);
		m_pipeLine.AddFilter(m_filter);
		
//		typedef M4D::Imaging::Image< uint16, 3 > VeiwImageType;
//		_convertor = new M4D::Imaging::ImageConvertor<VeiwImageType>();
//		_convertor->SetUpdateInvocationStyle(AbstractPipeFilter::UIS_ON_CHANGE_BEGIN);
//		m_pipeLine.AddFilter( _convertor );

		D_PRINT("Connecting recieved dataset into pipeline");
		// connect it to pipeline
		m_pipeLine.MakeInputConnection( *m_filter, 0, inputDataSet);
		
//		m_pipeLine.MakeConnection( *m_filter, 0, *_convertor, 0 );	

		// create and connect created output dataSet
		D_PRINT("Creating output connection")
		m_connWithOutputDataSet = &m_pipeLine.MakeOutputConnection( *m_filter, 0,
				true);
		
		

		// add message listener to be able catch execution done or failed messages
		ExecutionDoneCallback *callback = 
			new ExecutionDoneCallback(m_connWithOutputDataSet->GetDataset());
		MessageReceiverInterface::Ptr messasgeReciever = 
			MessageReceiverInterface::Ptr(callback);
		m_connWithOutputDataSet->SetMessageHook( messasgeReciever);
		
		m_pipeLine.ExecuteFirstFilter();
		
		// wait for execution done
		ScopedLock lock(callback->mutex);
		
		// wait for stuff within pipeline finish
		sleep(3000);
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
