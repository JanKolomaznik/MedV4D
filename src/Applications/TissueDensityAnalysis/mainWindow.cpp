#include "mainWindow.h"
#include "SettingsBox.h"
#include "m4dMySliceViewerWidget.h"
#include "Imaging/PipelineMessages.h"
#include "GUI/widgets/utils/ViewerFactory.h"
#include "ui_SettingsBox.h"


using namespace std;
using namespace M4D::Imaging;

typedef M4D::GUI::GenericViewerFactory< M4D::Viewer::m4dMySliceViewerWidget >	MySliceViewerFactory;


class LFNotifier : public M4D::Imaging::MessageReceiverInterface
{
public:
	LFNotifier( SphereSelectionFilter * filter ): _filter( filter ) {}
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
	SphereSelectionFilter * _filter;
};

void mainWindow::CreatePipeline()
{
	_convertor = new InImageConvertor();


	_filter = new SphereSelectionFilter();
	_filter->SetRadius(120);

	_pipeline.AddFilter(_convertor);
	_pipeline.AddFilter(_filter);

	_inConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeInputConnection( *_convertor, 0, false ) );
	//_inConnection = dynamic_cast<ConnectionInterfaceTyped<AbstractImage>*>( &_pipeline.MakeInputConnection( *_filter, 1, false ) );
	_pipeline.MakeConnection( *_convertor, 0, *_filter, 0 );
	_outConnection = dynamic_cast<ConnectionInterfaceTyped<AImage>*>( &_pipeline.MakeOutputConnection( *_filter, 0, true ) );

	if( _inConnection == NULL || _outConnection == NULL ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Pipeline error" ) );
	}

	addSource( _inConnection, "Simple MIP", "Input" );
	addSource( _outConnection, "Simple MIP", "Result" );

	_notifier =  new Notifier( this );
	_outConnection->SetMessageHook( MessageReceiverInterface::Ptr( _notifier ) );
}

void mainWindow::createDefaultViewerDesktop ()
{
  currentViewerDesktop = new M4D::GUI::m4dGUIMainViewerDesktopWidget( 1, 1, new MySliceViewerFactory() );
}

mainWindow::mainWindow ()
  : m4dGUIMainWindow(), _inConnection( NULL ), _outConnection( NULL )
{
}

void mainWindow::build(){
  M4D::GUI::m4dGUIMainWindow::build(APPLICATION_NAME, ORGANIZATION_NAME );

	Q_INIT_RESOURCE( mainWindow ); 

	CreatePipeline();

	_settings = new SettingsBox(this);
	_settings->build();

	 //M4D::Viewer::m4dMySliceViewerWidget *currentViewerWidget = reinterpret_cast<  M4D::Viewer::m4dMySliceViewerWidget *>(currentViewerDesktop->getSelectedViewerWidget());
	 M4D::Viewer::m4dGUIAbstractViewerWidget *currentViewerWidget = currentViewerDesktop->getSelectedViewerWidget();
   M4D::Viewer::m4dMySliceViewerWidget * v = (M4D::Viewer::m4dMySliceViewerWidget*)currentViewerWidget;

	 currentViewerWidget->setInputPort( _outConnection );

	 QObject::connect(_settings->ui->pushButton , SIGNAL( clicked() ), currentViewerWidget, SLOT( slotSetSpecialStateSelectMethodLeft() ),  Qt::QueuedConnection );
	 QObject::connect(v , SIGNAL( signalSphereCenter(double, double, double) ), _settings, SLOT( slotSetSphereCenter(double, double, double) ), Qt::DirectConnection );
	 //QObject::connect(v , SIGNAL( signalSphereRadius(int, int, double) ), _settings, SLOT( slotSetSphereRadius(int, int, double) ), Qt::DirectConnection );
	 QObject::connect( _settings->ui->lineEdit_x, SIGNAL( textChanged(QString)), _settings->ui->lineEdit_y, SLOT( setText(QString)), Qt::DirectConnection );

	addDockWindow( "Tissue Density Analysis", _settings );
	QObject::connect( _notifier, SIGNAL( Notification() ), _settings, SLOT( EndOfExecution() ), Qt::QueuedConnection );
}


void mainWindow::process ( ADataset::Ptr inputDataSet )
{
	try {
		_inConnection->PutDataset( inputDataSet );

		_convertor->Execute();

		currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0].UnPlug();
		_inConnection->ConnectConsumer( currentViewerDesktop->getSelectedViewerWidget()->InputPort()[0] );

		//_settings->SetEnabledExecButton( true );
		
	} 
	catch( ... ) {
		QMessageBox::critical( this, tr( "Exception" ), tr( "Some exception" ) );
	}
}


