#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"
#include "common/ExceptionBase.h"


using namespace std;
using namespace M4D::Imaging;

#define BUF_SIZE		256

#if WIN32
#define snprintf sprintf_s
#endif 


void mainWindow::CreatePipeline()
{

	_settings = new SettingsBox( currentViewerDesktop, _register, this );
	addDockWindow( "Volume Data Fusions", _settings );

	uint32 i;
	char buffer[ BUF_SIZE ];

	// create the connections, register filters and sources
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
		addSource( _inConnection[ i ], "Volume Data Fusions", buffer );
		
		snprintf( buffer, 255, "Transformed Image #%d", i+1 );
		addSource( _outConnection[ i ], "Volume Data Fusions", buffer );
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

	// set the dataset as the input of the selected viewer and as the input of the selected registration filter
	try {
		uint32 inputNumber = _settings->GetInputNumber();
		_inConnection[ inputNumber ]->PutDataset( inputDataSet );

		for ( uint32 i = 0; i < currentViewerDesktop->getSelectedViewerWidget()->InputPort().Size(); ++i ) currentViewerDesktop->getSelectedViewerWidget()->InputPort()[ i ].UnPlug();
		_inConnection[ inputNumber ]->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		if ( inputNumber == 0 )
		{
			_settings->ExecuteFilter( 0 );
			for ( uint32 i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i ) dynamic_cast< InImageRegistration* >( _register[ i ] )->SetReferenceImage( M4D::Imaging::AbstractImage::Cast( inputDataSet ) );
		}
	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}

void mainWindow::ClearDataset ()
{

	// clear the input of the selected connection and registration filter
	uint32 inputNumber = _settings->GetInputNumber();
	_inConnection[ inputNumber ]->ResetDataset();
	_register[ inputNumber ]->OutputPort().GetPort( 0 ).GetConnection()->ResetDataset();

	if ( inputNumber == 0 )
		for ( uint32 i = 0; i < SLICEVIEWER_INPUT_NUMBER; ++i ) dynamic_cast< InImageRegistration* >( _register[ i ] )->SetReferenceImage( M4D::Imaging::AbstractImage::Ptr() );

	currentViewerDesktop->UpdateViewers();
}

void mainWindow::OutConnectionToViewerPort( uint32 inputNumber, uint32 portNumber )
{
	if ( inputNumber >= SLICEVIEWER_INPUT_NUMBER ||
	     portNumber >= currentViewerDesktop->getSelectedViewerWidget()->InputPort().Size() )
	{
		_THROW_ M4D::ErrorHandling::EBadIndex();
	}

	// connect the given outconnection to the viewers input connection at the given position
	if ( _outConnection[ inputNumber ] != currentViewerDesktop->getSelectedViewerWidget()->InputPort()[ portNumber ].GetConnection() )
	{
		currentViewerDesktop->getSelectedViewerWidget()->InputPort()[ portNumber ].UnPlug();
		_outConnection[ inputNumber ]->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[ portNumber ] );
	}
}
