#include "mainWindow.h"

#include "Imaging/PipelineMessages.h"

#include "SettingsBox.h"

using namespace std;

using namespace M4D::Imaging;


class LFNotifier: public MessageReceiverInterface
{
  public:

	  LFNotifier ( APipeFilter * filter )
      : _filter( filter ) 
    {}
	  
    void ReceiveMessage ( PipelineMessage::Ptr msg, PipelineMessage::MessageSendStyle, FlowDirection )
	  {
		  if ( msg->msgID == PMI_FILTER_UPDATED ) {
			  _filter->ExecuteOnWhole();	
		  }
	  }

  protected:

	  APipeFilter * _filter;
};


void mainWindow::CreatePipeline ()
{
  convertor = new Convertor();
	_pipeline.AddFilter( convertor );

  registration = new Registration();
	_pipeline.AddFilter( registration );

	thresholding = new Thresholding();
	//_pipeline.AddFilter( thresholding );

	Median2D *median2D = new Median2D();

	median2D->SetUpdateInvocationStyle( APipeFilter::UIS_ON_CHANGE_BEGIN );
	median2D->SetRadius( 4 );
	//_pipeline.AddFilter( median2D );

	Mask *mask = new Mask();
	//_pipeline.AddFilter( mask );

	inConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeInputConnection( *convertor, 0, false ) );
	_pipeline.MakeConnection( *convertor, 0, *registration, 0 );
  //registrationTresholdConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeConnection( *registration, 0, *thresholding, 0 ) );
	//tresholdMedianConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeConnection( *thresholding, 0, *median2D, 0 ) );
	
	//_pipeline.MakeConnection( *registration, 0, *mask, 0 );

	//ConnectionInterface* medianMaskConnection = &(_pipeline.MakeConnection( *median2D, 0, *mask, 1 ) );
	//medianMaskConnection->SetMessageHook( MessageReceiverInterface::Ptr( new LFNotifier( mask ) ) );
	outConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeOutputConnection( *registration, 0, true ) );

	if( inConnection == NULL || outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( inConnection, "Perfusion Studies", "Input" );
  //addSource( registrationTresholdConnection, "Segmentation", "Registration - Treshold" );
	//addSource( tresholdMedianConnection, "Segmentation", "Treshold - Median" );
	//addSource( medianMaskConnection, "Segmentation", "Median - Mask" );
	addSource( outConnection, "Perfusion Studies", "Result" );

  _notifier = new Notifier(this);
	outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
}


mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME ), inConnection( NULL ), outConnection( NULL )
{
	Q_INIT_RESOURCE( mainWindow ); 

	CreatePipeline();

	_settings = new SettingsBox( registration, this );
	addDockWindow( "Perfusion Studies", _settings );
	QObject::connect( _notifier, SIGNAL(Notification()), _settings, SLOT(EndOfExecution()), Qt::QueuedConnection );
}


void mainWindow::process ( ADataset::Ptr inputDataSet )
{
	try {
		inConnection->PutDataset( inputDataSet );

		convertor->Execute();

		currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		inConnection->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_settings->SetEnabledExecButton( true );
	} 
	catch ( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}

