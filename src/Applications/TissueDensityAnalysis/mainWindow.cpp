#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"

using namespace std;
using namespace M4D::Imaging;

class LFNotifier : public M4D::Imaging::MessageReceiverInterface
{
public:
	LFNotifier( SphereSelectionFilter * filter ): _filter( filter ) {}
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
	SphereSelectionFilter * _filter;
};

void mainWindow::CreatePipeline()
{/*
	_convertor = new InImageConvertor();
	_pipeline.AddFilter( _convertor );

	_filter = new Thresholding();
	_pipeline.AddFilter( _filter );


	Median2D *medianFilter = new Median2D();
	medianFilter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
	medianFilter->SetRadius( 4 );
	_pipeline.AddFilter( medianFilter );

//
	MyFilter2D *myFilter = new MyFilter2D();
	myFilter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
	myFilter->SetRadius( 120 );
	_pipeline.AddFilter( myFilter );
//
	SphereSelectionFilter *sphereSelection = new SphereSelectionFilter();
	sphereSelection->SetRadius( 120 );
	_pipeline.AddFilter( sphereSelection );

	MaskSelectionFilter *maskSelection = new MaskSelectionFilter();
	_pipeline.AddFilter( maskSelection );

	_inConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeInputConnection( *_convertor, 0, false ) );
	_pipeline.MakeConnection( *_convertor, 0, *_filter, 0 );
	//_pipeline.MakeConnection( *_filter, 0, *medianFilter, 0 );
	_tmpConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeConnection( *_filter, 0, *medianFilter, 0 ) );
	
	_pipeline.MakeConnection( *_convertor, 0, *maskSelection, 0 );
	ConnectionInterface* tmpStage2 = &(_pipeline.MakeConnection( *medianFilter, 0, *maskSelection, 1 ) );
	tmpStage2->SetMessageHook( MessageReceiverInterface::Ptr( new LFNotifier( maskSelection ) ) );
	
	_pipeline.MakeConnection( *maskSelection, 0, *sphereSelection, 0 );
	_outConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeOutputConnection( *sphereSelection, 0, true ) );

	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( _inConnection, "Segmentation", "Input" );
	addSource( _tmpConnection, "Segmentation", "Stage #1" );
	addSource( tmpStage2, "Segmentation", "Stage #2" );
	addSource( _outConnection, "Segmentation", "Result" );

  _notifier = new Notifier(this);
	_outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
*/
	_convertor = new InImageConvertor();


	_filter = new SphereSelectionFilter();
	_filter->SetRadius( 120 );

	_pipeline.AddFilter( _convertor );
	_pipeline.AddFilter( _filter );

	_inConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeInputConnection( *_convertor, 0, false ) );
	_pipeline.MakeConnection( *_convertor, 0, *_filter, 0 );
	_outConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeOutputConnection( *_filter, 0, true ) );

	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( _inConnection, "Simple MIP", "Input" );
	addSource( _outConnection, "Simple MIP", "Result" );

	_notifier =  new Notifier( this );
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

void mainWindow::process ( AbstractDataSet::Ptr inputDataSet )
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

void mainWindow::switchToDefaultViewerDesktop ()
{
  switchToViewerDesktop( 2 );
}

