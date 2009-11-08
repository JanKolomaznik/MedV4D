#include "ViewerWindow.h"

ViewerWindow::ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage > & conn )
{
	viewerWidget = new M4D::Viewer::m4dGUIOpenGlViewerWidget( &conn, 0, NULL);
	//glWidget->setSelected( true );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::color_picker, M4D::Viewer::m4dGUIAbstractViewerWidget::right );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::adjust_bc, M4D::Viewer::m4dGUIAbstractViewerWidget::left );

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget((*viewerWidget)());
	setLayout(mainLayout);
	setFixedSize(800,800);

}

ViewerWindow::~ViewerWindow()
{} 