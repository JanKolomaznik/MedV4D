#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"

using namespace std;
using namespace M4D::Imaging;


mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME ), _inConnection( NULL ), _outConnection( NULL )
{
	Q_INIT_RESOURCE( mainWindow ); 

	CreatePipeline();

	// tell mainWindow about possible connections - can be during the creation of pipeline (connections)

	// M4D::Imaging::ConnectionInterface *conn;
	// addSource( conn, "Bone segmentation", "Stage #1" );
	// addSource( conn, "Bone segmentation", "Result" );

	// add your own settings widgets
	_settings = new SettingsBox( _filter, this );

	addDockWindow( "Bone Segmentation", _settings );

	QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
}


void 
mainWindow::process ( M4D::Dicom::DicomObjSetPtr dicomObjSet )
{
	AbstractImage::Ptr inputImage = M4D::Dicom::DcmProvider::CreateImageFromDICOM( dicomObjSet );

	try {
		_inConnection->PutDataset( inputImage );

		_convertor->Execute();

		mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_settings->SetEnabledExecButton( true );
	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}

}
class LFNotifier : public M4D::Imaging::MessageReceiverInterface
{
public:
	LFNotifier( AbstractPipeFilter * filter ): _filter( filter ) {}
	void
	ReceiveMessage( 
		M4D::Imaging::PipelineMessage::Ptr 			msg, 
		M4D::Imaging::PipelineMessage::MessageSendStyle 	/*sendStyle*/, 
		M4D::Imaging::FlowDirection				/*direction*/
		)
	{
		if( msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED ) 
		{
			_filter->ExecuteOnWhole();	
		}
	}

protected:
	AbstractPipeFilter * _filter;
};

void
mainWindow::CreatePipeline()
{
	_convertor = new InImageConvertor();
	_pipeline.AddFilter( _convertor );

	_filter = new Thresholding();
	_pipeline.AddFilter( _filter );

	Median2D *medianFilter = new Median2D();

	medianFilter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
	medianFilter->SetRadius( 4 );
	_pipeline.AddFilter( medianFilter );

	MaskSelectionFilter *maskSelection = new MaskSelectionFilter();
	//maskSelection->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
	_pipeline.AddFilter( maskSelection );

	_inConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeInputConnection( *_convertor, 0, false ) );
	_pipeline.MakeConnection( *_convertor, 0, *_filter, 0 );
	_tmpConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeConnection( *_filter, 0, *medianFilter, 0 ) );
	
	//_inConnection->ConnectConsumer( maskSelection->InputPort()[0] );
	_pipeline.MakeConnection( *_convertor, 0, *maskSelection, 0 );

	ConnectionInterface* tmpStage2 = &(_pipeline.MakeConnection( *medianFilter, 0, *maskSelection, 1 ) );
	tmpStage2->SetMessageHook( MessageReceiverInterface::Ptr( new LFNotifier( maskSelection ) ) );
	_outConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeOutputConnection( *maskSelection, 0, true ) );

	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( _inConnection, "Segmentation", "Input" );
	addSource( _tmpConnection, "Segmentation", "Stage #1" );
	addSource( tmpStage2, "Segmentation", "Stage #2" );
	addSource( _outConnection, "Segmentation", "Result" );

	_notifier =  new Notifier( this );
	_outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
}

