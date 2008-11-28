#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "GUI/m4dGUISliceViewerWidget.h"
#include "Imaging/ImageConnection.h"
#include "Imaging/Image.h"
#include "Common.h"


class ViewerWindow : public QWidget
{
private:
	M4D::Viewer::m4dGUIAbstractViewerWidget *viewerWidget;
public:
	ViewerWindow( M4D::Imaging::AbstractImageConnection& conn);
	~ViewerWindow();
};


ViewerWindow::ViewerWindow( M4D::Imaging::AbstractImageConnection& conn )
{
	viewerWidget = new M4D::Viewer::m4dGUISliceViewerWidget( &conn, 0, NULL );
	//glWidget->setSelected( true );

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget((*viewerWidget)());
	setLayout(mainLayout);
	setFixedSize(512,512);

}

ViewerWindow::~ViewerWindow()
{}

int
main( int argc, char** argv )
{
	if( argc < 2 || argc > 2 ) {
		std::cerr << "Wrong argument count - must be in form: 'program file'\n";
		return 1;
	}

	std::string filename = argv[1];


	std::cout << "Loading file...";
	M4D::Imaging::AbstractImage::AImagePtr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( filename );
	std::cout << "Done\n";
	

	M4D::Imaging::AbstractImageConnection prodconn;
	prodconn.PutImage( image );


	QApplication app(argc, argv);
	ViewerWindow viewer( prodconn );
	viewer.show();
	return app.exec();
}

