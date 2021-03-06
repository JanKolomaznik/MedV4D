#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"

using namespace std;
using namespace M4D::Imaging;

class LFNotifier : public M4D::Imaging::MessageReceiverInterface
{
public:
	LFNotifier( APipeFilter * filter ): _filter( filter ) {}
	void ReceiveMessage(M4D::Imaging::PipelineMessage::Ptr              msg, 
                      M4D::Imaging::PipelineMessage::MessageSendStyle /*sendStyle*/, 
                      M4D::Imaging::FlowDirection				              /*direction*/
		)
	{
		if( msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED ) {
			_filter->ExecuteOnWhole();	
		}
	}
protected:
	APipeFilter * _filter;
};

void mainWindow::CreatePipeline()
{
	_convertor = new InImageConvertor();
	_pipeline.AddFilter( _convertor );

	_filter = new Thresholding();
	_pipeline.AddFilter( _filter );

	Median2D *medianFilter = new Median2D();

	medianFilter->SetUpdateInvocationStyle( APipeFilter::UIS_ON_CHANGE_BEGIN );
	medianFilter->SetRadius( 4 );
	_pipeline.AddFilter( medianFilter );

	MaskSelectionFilter *maskSelection = new MaskSelectionFilter();
	_pipeline.AddFilter( maskSelection );

	_inConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeInputConnection( *_convertor, 0, false ) );
	_pipeline.MakeConnection( *_convertor, 0, *_filter, 0 );
	_tmpConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeConnection( *_filter, 0, *medianFilter, 0 ) );
	
	_pipeline.MakeConnection( *_convertor, 0, *maskSelection, 0 );

	ConnectionInterface* tmpStage2 = &(_pipeline.MakeConnection( *medianFilter, 0, *maskSelection, 1 ) );
	tmpStage2->SetMessageHook( MessageReceiverInterface::Ptr( new LFNotifier( maskSelection ) ) );
	_outConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeOutputConnection( *maskSelection, 0, true ) );

	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( _inConnection, "Segmentation", "Input" );
	addSource( _tmpConnection, "Segmentation", "Stage #1" );
	addSource( tmpStage2, "Segmentation", "Stage #2" );
	addSource( _outConnection, "Segmentation", "Result" );

  _notifier = new Notifier(this);
	_outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
}

mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME ), _inConnection( NULL ), _outConnection( NULL )
{
	Q_INIT_RESOURCE( mainWindow ); 

	CreatePipeline();

	_settings = new SettingsBox( _filter, this );
	addDockWindow( "Bone Segmentation", _settings );
	QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
}

void mainWindow::process ( ADataset::Ptr inputDataSet )
{
	try {
		_inConnection->PutDataset( inputDataSet );

		_convertor->Execute();

		currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_settings->SetEnabledExecButton( true );
	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}

