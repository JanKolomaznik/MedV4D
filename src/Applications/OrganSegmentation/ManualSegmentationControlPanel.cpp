#include "ManualSegmentationControlPanel.h"


ManualSegmentationControlPanel
::ManualSegmentationControlPanel( ManualSegmentationManager *manager ): QWidget(), _manager( manager )
{
	CreateWidgets();
}

void
ManualSegmentationControlPanel
::CreateWidgets()
{
	QVBoxLayout *verticalLayout;
	QStackedWidget *stackedWidget;
	QWidget *page;
	QWidget *verticalLayoutWidget_2;
	QVBoxLayout *verticalLayout_2;
	QWidget *page_2;
	QWidget *verticalLayoutWidget_3;
	QVBoxLayout *verticalLayout_3;
	QPushButton *button;

	verticalLayout = new QVBoxLayout();
	
	QToolBar *tbar = new QToolBar();
	_createSplineButton = tbar->addAction( "New\nSpline" );
	_createSplineButton->setCheckable( true );
	QObject::connect( _createSplineButton, SIGNAL( toggled( bool ) ), _manager, SLOT( SetCreatingState( bool ) ) );

	_editPointsButton = tbar->addAction( "Edit\nPoints" );
	_editPointsButton->setCheckable( true );
	QObject::connect( _editPointsButton, SIGNAL( toggled( bool ) ), _manager, SLOT( SetEditPointsState( bool ) ) );
	
	/*action = tbar->addAction( "Edit\nSegs" );
	action->setCheckable( true );*/

	verticalLayout->addWidget( tbar );
	
	/*stackedWidget = new QStackedWidget();
	page = new QWidget();
	verticalLayoutWidget_2 = new QWidget(page);
	verticalLayout_2 = new QVBoxLayout(verticalLayoutWidget_2);
	stackedWidget->addWidget(page);
	page_2 = new QWidget();
	verticalLayoutWidget_3 = new QWidget(page_2);
	verticalLayout_3 = new QVBoxLayout(verticalLayoutWidget_3);
	stackedWidget->addWidget(page_2);

	verticalLayout->addWidget(stackedWidget);*/

	_deleteCurveButton = new QPushButton( tr( "Delete Curve" ) );
	QObject::connect( _deleteCurveButton, SIGNAL(clicked()), _manager, SLOT( DeleteSelectedCurve()) );
	verticalLayout->addWidget( _deleteCurveButton );

	verticalLayout->addStretch( 3 );

	button = new QPushButton( tr( "Process Results" ) );
	QObject::connect( button, SIGNAL(clicked()), _manager, SLOT( WantProcessResults() ) );
	verticalLayout->addWidget( button );

	QObject::connect( _manager, SIGNAL(StateUpdated()), this, SLOT( PanelUpdate()) );

	verticalLayout->addStretch( 5 );

	setLayout(verticalLayout);
}

void
ManualSegmentationControlPanel
::PanelUpdate()
{
	switch( ManualSegmentationManager::Instance().GetInternalState() ) {
	case ManualSegmentationManager::SELECT:
		_deleteCurveButton->setEnabled( false );
		_createSplineButton->setEnabled( true );
		_editPointsButton->setEnabled( false );
		break;
	case ManualSegmentationManager::SELECTED:
		_deleteCurveButton->setEnabled( true );
		_createSplineButton->setEnabled( true );
		_editPointsButton->setEnabled( true );
		break;
	case ManualSegmentationManager::CREATING:
		_deleteCurveButton->setEnabled( false );
		_createSplineButton->setEnabled( true );
		_editPointsButton->setEnabled( false );
		break;
	case ManualSegmentationManager::SELECT_POINT:
		_deleteCurveButton->setEnabled( false );
		_createSplineButton->setEnabled( false );
		_editPointsButton->setEnabled( true );
		break;
	case ManualSegmentationManager::SELECTED_POINT:
		_deleteCurveButton->setEnabled( false );
		_createSplineButton->setEnabled( false );
		_editPointsButton->setEnabled( true );
		break;
	default:
		;
	}
}
