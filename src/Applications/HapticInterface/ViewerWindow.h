#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "m4dGUIOGLHapticViewerWidget.h"


class ViewerWindow : public QWidget
{
private:
	M4D::Viewer::m4dGUIOGLHapticViewerWidget *viewerWidget;
public:
	ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage > & conn);
	~ViewerWindow();
}; 