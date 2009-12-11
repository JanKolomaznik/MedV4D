#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "m4dGUIOGLHapticViewerWidget.h"
#include "..\..\gui\widgets\m4dGUISliceViewerWidget.h"


class ViewerWindow : public QWidget
{
private:
	M4D::Viewer::m4dGUIOGLHapticViewerWidget *viewerWidget;
	//M4D::Viewer::m4dGUISliceViewerWidget *viewerWidget;
public:
	ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > & conn);
	~ViewerWindow();
}; 