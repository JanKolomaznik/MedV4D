#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"

using namespace std;
using namespace M4D::Imaging;

#define BUF_SIZE		256

void mainWindow::CreatePipeline()
{

	_settings = new SettingsBox( currentViewerDesktop, _register, this );
	addDockWindow( "Volume Data Fusions", _settings );

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

		_notifier[ i ] = new Notifier(i, this);
		_outConnection[ i ]->SetMessageHook( MessageReceiverInterface::Ptr( _notifier[ i ] ) );

		QObject::connect( _notifier[ i ], SIGNAL( Notification(unsigned) ), _settings, SLOT( EndOfExecution(unsigned) ), Qt::QueuedConnection );

		snprintf( buffer, 255, "Input #%d", i+1 );
		addSource( _inConnection[ i ], "Segmentation", buffer );
		
		snprintf( buffer, 255, "Registered Image #%d", i+1 );
		addSource( _outConnection[ i ], "Segmentation", buffer );
	}


}

mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME )
{
	Q_INIT_RESOURCE( mainWindow ); 

	CreatePipeline();
}

void mainWindow::process ( AbstractDataSet::Ptr inputDataSet )
{
	try {
		uint32 inputNumber = _settings->GetInputNumber() - 1;
		_inConnection[ inputNumber ]->PutDataset( inputDataSet );

		for ( uint32 i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i ) currentViewerDesktop->getSelectedViewerWidget()->InputPort()[ i ].UnPlug();
		_inConnection[ inputNumber ]->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		if ( inputNumber == 0 )
			for ( uint32 i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i ) dynamic_cast< InImageRegistration* >( _register[ i ] )->SetReferenceImage( ImageType::Cast( inputDataSet ) );

	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}

