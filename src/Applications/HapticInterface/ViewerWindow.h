#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "m4dGUIOpenGLViewerWidget.h"


class ViewerWindow : public QWidget
{
private:
	M4D::Viewer::m4dGUIOpenGlViewerWidget *viewerWidget;
public:
	ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage > & conn);
	~ViewerWindow();
}; 