#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"

using namespace std;
using namespace M4D::Imaging;

#define BUF_SIZE		256

class LFNotifier : public M4D::Imaging::MessageReceiverInterface
{
public:
	LFNotifier( AbstractPipeFilter * filter ): _filter( filter ) {}
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
	AbstractPipeFilter * _filter;
};

void mainWindow::CreatePipeline()
{

	uint32 i;
	char buffer[ BUF_SIZE ];
	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i )
	{
		_register[ i ] = new InImageRegistration();
		_pipeline.AddFilter( _register[ i ] );
		_inConnection[ i ] = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeInputConnection( *_register[ i ], 0, false ) );
		_outConnection[ i ] = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeOutputConnection( *_register[ i ], 0, true ) );

		if( _inConnection[ i ] == NULL || _outConnection[ i ] == NULL ) {
			QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
		}

		snprintf( buffer, 255, "Input #%d", i+1 );
		addSource( _inConnection[ i ], "Segmentation", buffer );
		
		snprintf( buffer, 255, "Registered input #%d", i+1 );
		addSource( _outConnection[ i ], "Segmentation", buffer );
	}


/*
	_notifier = new Notifier(this);
	_outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
*/
}

mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME )
{
	Q_INIT_RESOURCE( mainWindow ); 

	CreatePipeline();

	_settings = new SettingsBox( currentViewerDesktop, this );
	addDockWindow( "Volume Data Fusions", _settings );
	//QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
}

void mainWindow::process ( AbstractDataSet::Ptr inputDataSet )
{
	try {
		uint32 inputNumber = _settings->GetInputNumber() - 1;
		_inConnection[ inputNumber ]->PutDataset( inputDataSet );

		currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection[ inputNumber ]->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		if ( inputNumber == 0 )
			for ( uint32 i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i ) dynamic_cast< InImageRegistration* >( _register[ i ] )->SetReferenceImage( ImageType::Cast( inputDataSet ) );

	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}

