#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_WINDOW
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_WINDOW

//////////////////////////////////////////////////////////////////////////
// Separate Window
//////////////////////////////////////////////////////////////////////////

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
	void closeEvent(QCloseEvent *event);
	M4D::Viewer::m4dGUIOGLHapticViewerWidget *viewerWidget;
}; 


//////////////////////////////////////////////////////////////////////////
// Medv4d GUI
//////////////////////////////////////////////////////////////////////////
//#include "MedV4D/GUI/widgets/m4dGUIMainWindow.h"
//#include "MedV4D/GUI/widgets/m4dGUIMainViewerDesktopWidget.h"
//
//#include "m4dGUIOGLHapticViewerWidget.h"
//
//#define ORGANIZATION_NAME     "MFF"
//#define APPLICATION_NAME      "Haptic Interface"
//
//class MainWindow: public M4D::GUI::m4dGUIMainWindow
//{
//	Q_OBJECT
//
//public:
//
//	void build();
//
//private:
//
//	virtual void createDefaultViewerDesktop ();
//};

#endif



