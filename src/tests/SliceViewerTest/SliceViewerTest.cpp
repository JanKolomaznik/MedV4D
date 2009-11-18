#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "GUI/widgets/GLSliceViewer.h"
#include "Imaging/Imaging.h"
#include "common/Common.h"


class ViewerWindow : public QWidget
{
private:
	M4D::Viewer::GLSliceViewer *viewerWidget;
public:
	ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage > & conn);
	~ViewerWindow();
};


ViewerWindow::ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage > & conn )
{
	viewerWidget = new M4D::Viewer::GLSliceViewer( NULL );
	conn.ConnectConsumer( viewerWidget->InputPort()[0] );
	//glWidget->setSelected( true );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::color_picker, M4D::Viewer::m4dGUIAbstractViewerWidget::right );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::adjust_bc, M4D::Viewer::m4dGUIAbstractViewerWidget::left );

	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget(viewerWidget->CastToQWidget());
	setLayout(mainLayout);
	setFixedSize(800,800);

}

ViewerWindow::~ViewerWindow()
{}

int
main( int argc, char** argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	if( argc < 2 || argc > 2 ) {
		std::cerr << "Wrong argument count - must be in form: 'program file'\n";
		return 1;
	}

	std::string filename = argv[1];


	std::cout << "Loading file...";
	M4D::Imaging::AbstractImage::Ptr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( filename );
	std::cout << "Done\n";
	

	std::cout << "Add dataset to connection...";
	M4D::Imaging::ConnectionTyped< M4D::Imaging::AbstractImage > prodconn;
	prodconn.PutDataset( image );
	std::cout << "Done\n";


	QApplication app(argc, argv);
	ViewerWindow viewer( prodconn );
	viewer.show();
	return app.exec();
}

