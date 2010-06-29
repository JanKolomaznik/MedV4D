#include "ViewerWindow.h"


//////////////////////////////////////////////////////////////////////////
// Medv4d GUI
//////////////////////////////////////////////////////////////////////////



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
	setFixedSize(1300,800);
	build();
}

ViewerWindow::~ViewerWindow()
{} 
void ViewerWindow::build()
{
}


//////////////////////////////////////////////////////////////////////////
// MedV4d GUI
//////////////////////////////////////////////////////////////////////////

//void MainWindow::build()
//{
//	m4dGUIMainWindow::build(APPLICATION_NAME, ORGANIZATION_NAME);
//
//	Q_INIT_RESOURCE( mainWindow );
//}
//
//void MainWindow::createDefaultViewerDesktop()
//{
//	currentViewerDesktop = new M4D::GUI::m4dGUIMainViewerDesktopWidget( 1, 1, new M4D::Viewer::OGLHapticViewerFactory );
//}