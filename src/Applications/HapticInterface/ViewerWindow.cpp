#include "ViewerWindow.h"
#include "ui_SettingsBox.h"


ViewerWindow::ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > & conn )
{
	viewerWidget = new M4D::Viewer::m4dGUIOGLHapticViewerWidget( &conn, 0, NULL);
	//viewerWidget = new M4D::Viewer::m4dGUISliceViewerWidget( &conn, 0, NULL);
	//glWidget->setSelected( true );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::color_picker, M4D::Viewer::m4dGUIAbstractViewerWidget::right );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::adjust_bc, M4D::Viewer::m4dGUIAbstractViewerWidget::left );

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget((*viewerWidget)());
	setLayout(mainLayout);
	setFixedSize(1024,800);
	build();
}

ViewerWindow::~ViewerWindow()
{} 
void ViewerWindow::build()
{
	settings = new SettingsBox(this);

	M4D::Viewer::m4dGUIAbstractViewerWidget *currentViewerWidget = viewerWidget;

	QObject::connect(settings->ui->steSizeButton, SIGNAL( clicked() ), settings, SLOT( slotChangeScale() ), Qt::QueuedConnection );
	QObject::connect(settings, SIGNAL( scaleChanged(double) ), currentViewerWidget, SLOT( slotSetScale(double) ), Qt::DirectConnection );

	// zadockuj me
	//settings->
	settings->show();
}