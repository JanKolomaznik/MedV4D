#include "MainWindow.h"

#include "Imaging/PipelineMessages.h"

#include "SettingsBox.h"
#include "PlotBox.h"

using namespace std;

using namespace M4D::Imaging;


MainWindow::MainWindow ()
  : m4dGUIMainWindow( APPLICATION_NAME, ORGANIZATION_NAME, QIcon( ":/resources/parameter.png" ) ), inConnection( NULL )
{
	Q_INIT_RESOURCE( mainWindow ); 

  for ( uint8 i = 0; i < outConnection.size(); i++ ) { 
    outConnection[i] = NULL;
  }

  CreatePipeline();

	settings = new SettingsBox( registration, segmentation, analysis, this );
	QDockWidget *settingsDock = addDockWindow( "Perfusion Studies", settings );
  settingsDock->move( x() + width() - settingsDock->width() - 30, y() + 135 );

	connect( notifier, SIGNAL(Notification()), settings, SLOT(EndOfExecution()), Qt::QueuedConnection );
  connect( settings, SIGNAL(VisualizationDone()), currentViewerDesktop, SLOT(UpdateViewers()) );
  connect( settings, SIGNAL(SimpleSelected()), this, SLOT(SetSelectedViewerToSimple()) );
  connect( settings, SIGNAL(ParamaterMapsSelected()), this, SLOT(SetSelectedViewerToRGB()) );
  connect( settings, SIGNAL(CurveToolSelected( bool )), this, SLOT(SetSelectedViewerToPoint( bool )) );
  connect( settings, SIGNAL(CutToolSelected( bool )), this, SLOT(SetSelectedViewerToRegion( bool )) );

  plot = new PlotBox( analysis, this );
  QDockWidget *plotDock = addDockWindow( "TIC plot", plot );
  plotDock->move( x() + 30, y() + 135 );
  plotDock->hide();

  connect( settings, SIGNAL(CurveToolSelected( bool )), plotDock, SLOT(setVisible( bool )) );
  
  connect( currentViewerDesktop, SIGNAL(sourceChanged()), this, SLOT(SourceSelected()) );
}


void MainWindow::CreatePipeline ()
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


void MainWindow::process ( ADataset::Ptr inputDataSet )
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


void MainWindow::SetSelectedViewerToSimple ()
{
  vector< M4D::Viewer::m4dGUIAbstractViewerWidget * > viewers;
  currentViewerDesktop->getViewerWidgetsWithSource( SOURCE_NUMBER - 1, viewers );

  // loop over all viewers connected to the output
  for ( uint8 i = 0; i < viewers.size(); i++ ) {
    dynamic_cast< M4D::Viewer::m4dGUISliceViewerWidget * >( viewers[i] )->setTexturePreparerToSimple();
  }
}


void MainWindow::SetSelectedViewerToRGB ()
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


void MainWindow::SetSelectedViewerToPoint ( bool toolEnabled )
{
  vector< M4D::Viewer::m4dGUIAbstractViewerWidget * > viewers;
  currentViewerDesktop->getViewerWidgetsWithSource( SOURCE_NUMBER - 1, viewers );

  // loop over all viewers connected to the output
  for ( uint8 i = 0; i < viewers.size(); i++ ) 
  {
    if ( toolEnabled ) 
    {
      viewers[i]->slotSetButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::point_picker, M4D::Viewer::m4dGUIAbstractViewerWidget::left );
      connect( viewers[i], SIGNAL(signalDataPointPicker( unsigned, int, int, int)), plot, SLOT(pointPicked( unsigned, int, int, int)) );
    } 
    else {
      disconnect( viewers[i], SIGNAL(signalDataPointPicker( unsigned, int, int, int)), plot, SLOT(pointPicked( unsigned, int, int, int)) );
    }
  }
}


void MainWindow::SetSelectedViewerToRegion ( bool toolEnabled )
{
  vector< M4D::Viewer::m4dGUIAbstractViewerWidget * > viewers;
  currentViewerDesktop->getViewerWidgetsWithSource( SOURCE_NUMBER - 1, viewers );

  // loop over all viewers connected to the output
  for ( uint8 i = 0; i < viewers.size(); i++ ) 
  {
    if ( toolEnabled ) {
      viewers[i]->slotSetButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::region_picker, M4D::Viewer::m4dGUIAbstractViewerWidget::left );
    } 
    else 
    {
      viewers[i]->slotSetButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::point_picker, M4D::Viewer::m4dGUIAbstractViewerWidget::left );
      texturePreparer.setLastClickedPosition( -1, -1, -1 );
    }
  }
}


void MainWindow::SourceSelected ()
{
  if ( static_cast< Analysis * >( analysis )->GetVisualizationType() == VT_PARAM ) 
  {
    M4D::Viewer::m4dGUIAbstractViewerWidget *viewer;
    viewer = currentViewerDesktop->getSelectedViewerWidget();

    texturePreparer.setMinMaxValue( 0, static_cast< Analysis * >( analysis )->GetMaxParameterValue() );

    dynamic_cast< M4D::Viewer::m4dGUISliceViewerWidget * >( viewer )->setTexturePreparerToCustom( &texturePreparer );
  }
}
