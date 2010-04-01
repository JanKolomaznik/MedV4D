//#include "Imaging/ImageFactory.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <QtGui/QApplication>
#include <QtOpenGL/QGLWidget>
#include <QtGui/QWidget>
#include <QtGui/QHBoxLayout>
#include "GUI/widgets/GLThreadedWidget.h"
#include "Sample3DViewer.h"
#include "common/Common.h"

#include "common/Quaternion.h"

#include "Imaging/Mesh.h"

#ifdef Q_WS_X11
#include <X11/Xlib.h>  // for XInitThreads() call
#endif

using namespace M4D::Imaging;
using namespace M4D::Imaging::Geometry;

//typedef Sample3DViewer<GLThreadedWidget> Viewer;
typedef Sample3DViewer<QGLWidget> Viewer;


class ViewerWindow : public QWidget
{
private:
	Viewer *viewerWidget;

public:
	ViewerWindow();
	~ViewerWindow();
};


ViewerWindow::ViewerWindow(): QWidget( NULL )
{
	viewerWidget = new Viewer( NULL );

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget(viewerWidget);
	setLayout(mainLayout);
	resize(800,800);

}

ViewerWindow::~ViewerWindow()
{}

int
main( int argc, char** argv )
{
#ifdef Q_WS_X11
    // this needs to be the first in the app to make Xlib calls thread save
    // needed for OpenGl rendering threads
    XInitThreads();
#endif

	/*std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );*/

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );
	
	QApplication app(argc, argv);
	ViewerWindow viewer;
	viewer.show();
	return app.exec();
}

