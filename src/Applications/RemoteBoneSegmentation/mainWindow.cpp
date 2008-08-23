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

  _settings = new SettingsBox( _filter );

  addDockWindow( "Remote bone Segmentation", _settings );

  QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
}


void
mainWindow::process ( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet )
{
  AbstractImage::AImagePtr inputImage = ImageFactory::CreateImageFromDICOM( dicomObjSet );


  unsigned dim = inputImage->GetDimension();
  int type     = inputImage->GetElementTypeID();

  if ( dim != 3 || type != GetNumericTypeID<ElementType>() ) {
    //TODO throw exception

    QMessageBox::critical( this, tr( "Exception" ), tr( "Bad type" ) );
    return;
  }
  try {
    _inConnection->PutImage( inputImage );

    /*_inConnection->RouteMessage( MsgFilterUpdated::CreateMsg( true ),
        PipelineMessage::MSS_NORMAL,
        FD_IN_FLOW
        );*/

    //mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
    //conn->ConnectConsumer( mainViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );
  }
  catch( ... ) {
    QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
  }

}

void
mainWindow::CreatePipeline()
{
  _filter = new Thresholding();
  Median2D *tmpFilter = new Median2D();
  tmpFilter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_CHANGE_BEGIN );
  //tmpFilter->SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_UPDATE_FINISHED );

  tmpFilter->SetRadius( 2 );

  _pipeline.AddFilter( _filter );
  _pipeline.AddFilter( tmpFilter );
  ;

  _inConnection = dynamic_cast<AbstractImageConnectionInterface*>( &_pipeline.MakeInputConnection( *_filter, 0, false ) );
  _tmpConnection = dynamic_cast<AbstractImageConnectionInterface*>( &_pipeline.MakeConnection( *_filter, 0, *tmpFilter, 0 ) );
  _outConnection = dynamic_cast<AbstractImageConnectionInterface*>( &_pipeline.MakeOutputConnection( *tmpFilter, 0, true ) );

  if( _inConnection == NULL || _outConnection == NULL ) {
    QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
  }

  addSource( _inConnection, "Remote bone segmentation", "Input" );
  addSource( _tmpConnection, "Remote bone segmentation", "Stage #1" );
  addSource( _outConnection, "Remote bone segmentation", "Result" );

  _notifier =  new Notifier( this );
  _outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
}

