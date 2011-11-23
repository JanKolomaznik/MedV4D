#include "mainWindow.h"
#include "SettingsBox.h"
#include "TDASliceViewerWidget.h"
#include "ui_SettingsBox.h"

using namespace std;
using namespace M4D::Imaging;
using namespace M4D::GUI;


typedef GenericViewerFactory< TDASliceViewerWidget >	MySliceViewerFactory;


mainWindow::mainWindow ()
  : m4dGUIMainWindow(), _inConnection( NULL ), _outConnection( NULL )
{
}

void mainWindow::createDefaultViewerDesktop ()
{
	// nahradim defaultny viewer vlastni viewrem MySliceViewerWidget
	currentViewerDesktop = new m4dGUIMainViewerDesktopWidget( 2, 2, new MySliceViewerFactory() );
}


void mainWindow::build(){
	M4D::GUI::m4dGUIMainWindow::build(APPLICATION_NAME, ORGANIZATION_NAME );

	Q_INIT_RESOURCE( mainWindow ); 

	CreatePipeline();
	

	_settings = new SettingsBox(this);
	
	_settings->build();
	_settings->setMaskFilter(_filter);

	 M4D::Viewer::m4dGUIAbstractViewerWidget *currentViewerWidget = currentViewerDesktop->getSelectedViewerWidget();
	 TDASliceViewerWidget * v = (TDASliceViewerWidget*)currentViewerWidget;

	 currentViewerWidget->setInputPort( _outConnection );

	 QObject::connect(_settings->ui->pushButton , SIGNAL( clicked() ), currentViewerWidget, SLOT( slotSetSpecialStateSelectMethodLeft() ),  Qt::QueuedConnection );
	 QObject::connect(v , SIGNAL( signalSphereCenter(double, double, double) ), _settings, SLOT( slotSetSphereCenter(double, double, double) ), Qt::DirectConnection );
	 //QObject::connect(v , SIGNAL( signalSphereRadius(int, int, double) ), _settings, SLOT( slotSetSphereRadius(int, int, double) ), Qt::DirectConnection );
	 QObject::connect( _settings->ui->lineEdit_x, SIGNAL( textChanged(QString)), _settings->ui->lineEdit_y, SLOT( setText(QString)), Qt::DirectConnection );
	 QObject::connect(_settings->ui->pushButton_3 , SIGNAL( clicked() ), _settings, SLOT( slotCreateMask() ),  Qt::QueuedConnection );

	addDockWindow( "Tissue Density Analysis", _settings );
	QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
}



void mainWindow::process ( ADataset::Ptr inputDataSet )
{
	try {
		_inConnection->PutDataset( inputDataSet );

		_convertor->Execute();
		
		for ( unsigned i = 0; i < currentViewerDesktop->getSelectedViewerWidget()->InputPort().Size(); i++ ) {
		  currentViewerDesktop->getSelectedViewerWidget()->InputPort()[i].UnPlug();
		}

		_inMaskConnection->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );
		_outConnection->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[1] );

		//_settings->SetEnabledExecButton( true );
		
	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}


void mainWindow::CreatePipeline()
{
	_convertor = new InImageConvertor();

	_filter = new SphereSelectionFilter();
	_filter->SetRadius(120);
	_filter->SetColumnCenter(200);
    _filter->SetUpdateInvocationStyle(APipeFilter::UIS_ON_UPDATE_FINISHED);

	_pipeline.AddFilter(_convertor);
	_pipeline.AddFilter(_filter);
	_filter->SetColumnCenter(200);

	_inConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeInputConnection( *_convertor, 0, false ) );
	_inMaskConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeConnection( *_convertor, 0, *_filter, 0 ) );
	_outConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeOutputConnection( *_filter, 0, true ) );

	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( _inConnection, "Simple MIP", "Input" );
	addSource( _outConnection, "Simple MIP", "Result" );


	 M4D::Viewer::m4dGUIAbstractViewerWidget *currentViewerWidget = currentViewerDesktop->getSelectedViewerWidget();
	 TDASliceViewerWidget *v = (TDASliceViewerWidget*)currentViewerWidget;

	 v->setMaskConnection(_outConnection);

	_notifier =  new Notifier( this );
	_outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
}