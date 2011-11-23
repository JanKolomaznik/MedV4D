#include "MedV4D/Common/Common.h"
#include "Imaging/PipelineMessages.h"
//#include "Imaging/interpolators/nearestNeighbor.h"
#include "Imaging/filters/imageSizeAdapter.h"
#include "mainWindow.h"
#include "SettingsBox.h"

using namespace std;
using namespace M4D::Imaging;

#define SETTINGS_WINDOW_NAME "Algorithm settigs"

mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME )
  , _filter( NULL )
  , _inConnection( NULL )
  , _outConnection( NULL )  
{
	Q_INIT_RESOURCE( mainWindow ); 

	CreatePipeline();

	// tell mainWindow about possible connections - can be during the creation of pipeline (connections)

	// M4D::Imaging::ConnectionInterface *conn;
	// addSource( conn, "Bone segmentation", "Stage #1" );
	// addSource( conn, "Bone segmentation", "Result" );

	// add your own settings widgets
	_settings = new SettingsBox( (RemoteFilterType *)_filter, &properties_, this, _tmpConnection->GetDataset() );

	addDockWindow( SETTINGS_WINDOW_NAME, _settings, DOCKED_DOCK_WINDOW);

	QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
	
	QObject::connect( _adapterDoneNotifier, SIGNAL( Notification() ), this, SLOT( OnAdapterDone() ), Qt::QueuedConnection );
}

void
mainWindow::OnAdapterDone()
{
	Vector<uint32, 3> dsSize;
	Image<float32, 3> &im = 
		static_cast<Image<float32, 3> &>(_tmpConnection->GetDataset());
	
	for(uint32 i=0; i<im.Dimension; i++)
	{
		dsSize[i] =	
			(im.GetDimensionExtents(i).maximum - im.GetDimensionExtents(i).minimum) / 2;
	}
	_settings->SetSeed(dsSize);
	
	_settings->SetEnabledExecButton( true );
}

void 
mainWindow::process ( AbstractDataSet::Ptr inputDataSet )
{
	try {
		_settings->SetEnabledExecButton( false );
//
//				currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
//				conn->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_inConnection->PutDataset( inputDataSet );

		_convertor->Execute();

		currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}

}

void
mainWindow::CreatePipeline()
{
	// Create filters
	_convertor = new ViewImageConvertor();
	_pipeline.AddFilter( _convertor );
	
	_decimatedImConvertor = new ViewImageConvertor();
	_decimatedImConvertor->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
	_pipeline.AddFilter( _decimatedImConvertor );
	
	_resultImConvertor = new ViewImageConvertor();
	_resultImConvertor->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_UPDATE_FINISHED );
	_pipeline.AddFilter( _resultImConvertor );
	
	typedef ImageSizeAdapter<VeiwImageType> ForCellSizeAdapter;
	
#define DESIRED_IMAGE_SIZE_FOR_CELL (20*1024*1024) // 20M
	_decimator = new ForCellSizeAdapter( 
			new ForCellSizeAdapter::Properties(DESIRED_IMAGE_SIZE_FOR_CELL) );
	_decimator->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
	_pipeline.AddFilter( _decimator );
	
	_filter = new RemoteFilterType(& properties_);
	_pipeline.AddFilter( _filter );
	
	
	// create connections
	_inConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( 
			&_pipeline.MakeInputConnection( *_convertor, 0, false ) );
	_pipeline.MakeConnection( *_convertor, 0, *_decimator, 0 );
	
	_tmpConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( 
			&_pipeline.MakeConnection( *_decimator, 0, *_filter, 0 ) );
	
	_remote2castConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( 
				&_pipeline.MakeConnection( *_filter, 0, *_resultImConvertor, 0 ) );	
	
	_outConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( 
			&_pipeline.MakeOutputConnection( *_resultImConvertor, 0, true ) );
	
	_decim2castConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( 
					&_pipeline.MakeConnection( *_decimator, 0, *_decimatedImConvertor, 0 ) );
	
	_castOutConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( 
			&_pipeline.MakeOutputConnection( *_decimatedImConvertor, 0, true ) );
	
	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( _inConnection, SETTINGS_WINDOW_NAME, "Input" );
	addSource( _castOutConnection, SETTINGS_WINDOW_NAME, "Decimated image" );
	addSource( _outConnection, SETTINGS_WINDOW_NAME, "Result" );

	_notifier =  new Notifier( this );
	_outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
	
	_adapterDoneNotifier = new Notifier(this);
	_tmpConnection->SetMessageHook(
			MessageReceiverInterface::Ptr(_adapterDoneNotifier));
}

