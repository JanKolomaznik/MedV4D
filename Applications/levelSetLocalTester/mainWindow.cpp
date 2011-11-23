#include "mainWindow.h"
#include "SettingsBox.h"
#include "Imaging/PipelineMessages.h"

//#include "itkIntegration/dicomReader.h"
//#include "remoteComp/remoteServerFilters/decimatorFilter/decimatorFilter.h"
#include "Imaging/interpolators/nearestNeighbor.h"
#include "Imaging/filters/decimation.h"

using namespace std;
using namespace M4D::Imaging;


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
	_settings = new SettingsBox( (RemoteFilterType *)_filter, &properties_, this );

	addDockWindow( "Simple MIP", _settings, DOCKED_DOCK_WINDOW );

	QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
}


void 
mainWindow::process ( M4D::Dicom::DicomObjSetPtr dicomObjSet )
{
	try {
		AbstractImage::Ptr inputImage = M4D::Dicom::DcmProvider::CreateImageFromDICOM( dicomObjSet );
//		typedef Image<int16, 3> MedevedImage;
//		AbstractImage::Ptr inputImage = ImageFactory::CreateEmptyImage3D<int16>(1, 1, 1);	// buddy
//		
//		M4D::ITKIntegration::ReadImage::ReadImage(
//			M4D::ITKIntegration::ReadImage,
//			"/home/vencax/data/pok");

		_inConnection->PutDataset( inputImage );

		_convertor->Execute();

		currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		_settings->SetEnabledExecButton( true );

	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}

}

void
mainWindow::CreatePipeline()
{
	_convertor = new InImageConvertor();
	_pipeline.AddFilter( _convertor );
	
	typedef M4D::Imaging::NearestNeighborInterpolator<ImageType> Interpolator;
	typedef M4D::Imaging::DecimationFilter<ImageType, Interpolator> Decimator;
	
	
	_decimator = new Decimator( new Decimator::Properties(0.5f) );
	_decimator->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
	_pipeline.AddFilter( _decimator );
	
	_filter = new RemoteFilterType(& properties_);
	_pipeline.AddFilter( _filter );

	_inConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( 
			&_pipeline.MakeInputConnection( *_convertor, 0, false ) );
	_pipeline.MakeConnection( *_convertor, 0, *_decimator, 0 );	
//		_pipeline.MakeConnection( *_convertor, 0, *_filter, 0 );
	
	_tmpConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( 
			&_pipeline.MakeConnection( *_decimator, 0, *_filter, 0 ) );
	
	_outConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( 
			&_pipeline.MakeOutputConnection( *_filter, 0, true ) );
	
	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( _inConnection, "Simple MIP", "Input" );
	addSource( _tmpConnection, "Simple MIP", "Decimated image" );
	addSource( _outConnection, "Simple MIP", "Result" );

	_notifier =  new Notifier( this );
	_outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
}

