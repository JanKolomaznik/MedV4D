#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_WINDOW
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_WINDOW

#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include <QMainWindow>
#include "m4dGUIOGLHapticViewerWidget.h"

class SettingsBox;

class ViewerWindow : public QWidget
{
public:
	ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > & conn);
	~ViewerWindow();
	void build();
private:
	M4D::Viewer::m4dGUIOGLHapticViewerWidget *viewerWidget;
}; 

#endif
