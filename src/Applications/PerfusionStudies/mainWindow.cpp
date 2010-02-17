#include "mainWindow.h"

#include "Imaging/PipelineMessages.h"

#include "SettingsBox.h"

using namespace std;

using namespace M4D::Imaging;


mainWindow::mainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME, QIcon( ":/resources/parameter.png" ) ), inConnection( NULL )
{
	Q_INIT_RESOURCE( mainWindow ); 

  for ( uint8 i = 0; i < outConnection.size(); i++ ) { 
    outConnection[i] = NULL;
  }

  createPipeline();

	settings = new SettingsBox( registration, segmentation, analysis, this );
	addDockWindow( "Perfusion Studies", settings );
	connect( notifier, SIGNAL(Notification()), settings, SLOT(EndOfExecution()), Qt::QueuedConnection );
  connect( settings, SIGNAL(VisualizationDone()), currentViewerDesktop, SLOT(UpdateViewers()) );
  connect( settings, SIGNAL(SimpleSelected()), this, SLOT(setSelectedViewerToSimple()) );
  connect( settings, SIGNAL(ParamaterMapsSelected()), this, SLOT(setSelectedViewerToRGB()) );
  
  connect( currentViewerDesktop, SIGNAL(sourceChanged()), this, SLOT(sourceSelected()) );
}


void mainWindow::createPipeline ()
{
  convertor = new Convertor();
	pipeline.AddFilter( convertor );

  registration = new Registration();
	pipeline.AddFilter( registration );

	segmentation = new Segmentation();
  segmentation->SetUpdateInvocationStyle( APipeFilter::UIS_ON_CHANGE_BEGIN );
	pipeline.AddFilter( segmentation );

  analysis = new Analysis();
  analysis->SetUpdateInvocationStyle( APipeFilter::UIS_ON_CHANGE_BEGIN );
	pipeline.AddFilter( analysis );

	inConnection = dynamic_cast< ConnectionType * >( &pipeline.MakeInputConnection( *convertor, 0, false ) );

	pipeline.MakeConnection( *convertor, 0, *registration, 0 );
  registrationSegmentationConnection = dynamic_cast< ConnectionType * >( &pipeline.MakeConnection( *registration, 0, *segmentation, 0 ) );
  segmentationAnalysisConnection = dynamic_cast< ConnectionType * >( &pipeline.MakeConnection( *segmentation, 0, *analysis, 0 ) );
		
  for ( uint8 i = 0; i < 3; i++ ) {
    outConnection.push_back( dynamic_cast< ConnectionType * >( &pipeline.MakeOutputConnection( *analysis, i, true ) ) );
  }

	if( inConnection == NULL || outConnection.empty() ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( inConnection, "Perfusion Studies", "Input" );
  addSource( registrationSegmentationConnection, "Perfusion Studies", "Registration - Segmentation" );
  addSource( segmentationAnalysisConnection, "Perfusion Studies", "Segmentation - Analysis" );
	addSource( outConnection, "Perfusion Studies", "Result" );

  notifier = new Notifier( this );
	outConnection[0]->SetMessageHook( MessageReceiverInterface::Ptr( notifier ) );
}


void mainWindow::process ( ADataset::Ptr inputDataSet )
{
	try 
  {
		inConnection->PutDataset( inputDataSet );

		convertor->Execute();

    vector< M4D::Viewer::m4dGUIAbstractViewerWidget * > viewers;
    currentViewerDesktop->getViewerWidgetsWithSource( 0, viewers );

    // loop over all viewers connected to the input
    for ( uint8 i = 0; i < viewers.size(); i++ ) 
    {
		  viewers[i]->InputPort()[0].UnPlug();
		  inConnection->ConnectConsumer( viewers[i]->InputPort()[0] );      
      
      dynamic_cast< M4D::Viewer::m4dGUISliceViewerWidget * >( viewers[i] )->setTexturePreparerToSimple();
    }

    settings->SetEnabledExecButton( true );
	} 
	catch ( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}


void mainWindow::setSelectedViewerToSimple ()
{
  vector< M4D::Viewer::m4dGUIAbstractViewerWidget * > viewers;
  currentViewerDesktop->getViewerWidgetsWithSource( SOURCE_NUMBER - 1, viewers );

  // loop over all viewers connected to the output
  for ( uint8 i = 0; i < viewers.size(); i++ ) {
    dynamic_cast< M4D::Viewer::m4dGUISliceViewerWidget * >( viewers[i] )->setTexturePreparerToSimple();
  }
}


void mainWindow::setSelectedViewerToRGB ()
{
  vector< M4D::Viewer::m4dGUIAbstractViewerWidget * > viewers;
  currentViewerDesktop->getViewerWidgetsWithSource( SOURCE_NUMBER - 1, viewers );

  texturePreparer.setMinMaxValue( 0, static_cast< Analysis * >( analysis )->GetMaxParameterValue() );

  // loop over all viewers connected to the output
  for ( uint8 i = 0; i < viewers.size(); i++ ) 
  {
    dynamic_cast< M4D::Viewer::m4dGUISliceViewerWidget * >( viewers[i] )->setTexturePreparerToCustom( &texturePreparer );
    viewers[i]->updateViewer();
  }
}


void mainWindow::sourceSelected ()
{
  if ( static_cast< Analysis * >( analysis )->GetVisualizationType() == VT_PARAM ) 
  {
    M4D::Viewer::m4dGUIAbstractViewerWidget *viewer;
    viewer = currentViewerDesktop->getSelectedViewerWidget();

    texturePreparer.setMinMaxValue( 0, static_cast< Analysis * >( analysis )->GetMaxParameterValue() );

    dynamic_cast< M4D::Viewer::m4dGUISliceViewerWidget * >( viewer )->setTexturePreparerToCustom( &texturePreparer );
  }
}
