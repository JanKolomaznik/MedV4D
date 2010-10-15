#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "GUI/widgets/BasicSliceViewer.h"
#include "Imaging/Imaging.h"
#include "common/Common.h"
#include "ViewerWindow.hpp"

#include <tclap/CmdLine.h>

/*class ViewerWindow : public QWidget
{
private:
	M4D::GUI::Viewer::BasicSliceViewer *viewerWidget;
public:
	ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > & conn);
	//ViewerWindow( M4D::Imaging::AImage::Ptr image );
	~ViewerWindow();
};


ViewerWindow::ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > & conn )
{
	viewerWidget = new M4D::GUI::Viewer::BasicSliceViewer();

	conn.ConnectConsumer( viewerWidget->InputPort()[0] );
	//glWidget->setSelected( true );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::color_picker, M4D::Viewer::m4dGUIAbstractViewerWidget::right );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::adjust_bc, M4D::Viewer::m4dGUIAbstractViewerWidget::left );

	viewerWidget->ZoomFit();
	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget(viewerWidget->CastToQWidget());
	setLayout(mainLayout);
	resize(600,600);

}*/

/*ViewerWindow::ViewerWindow( M4D::Imaging::AImage::Ptr image )
{
	viewerWidget = new M4D::GUI::Viewer::BasicSliceViewer();

	//conn.ConnectConsumer( viewerWidget->InputPort()[0] );
	//glWidget->setSelected( true );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::color_picker, M4D::Viewer::m4dGUIAbstractViewerWidget::right );
	//viewerWidget->setButtonHandler( M4D::Viewer::m4dGUIAbstractViewerWidget::adjust_bc, M4D::Viewer::m4dGUIAbstractViewerWidget::left );

	viewerWidget->SetImage( image );

	viewerWidget->ZoomFit();
	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget(viewerWidget->CastToQWidget());
	setLayout(mainLayout);
	resize(600,600);

}*/

/*ViewerWindow::~ViewerWindow()
{}
*/

std::string inFilename;

void
processCommandLine( int argc, char** argv )
{
	TCLAP::CmdLine cmd( "Median filter.", ' ', "");
	/*---------------------------------------------------------------------*/

	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "filename1" );
	cmd.add( inFilenameArg );

	cmd.parse( argc, argv );

	inFilename = inFilenameArg.getValue();
}

int
main( int argc, char** argv )
{
	//std::ofstream logFile( "Log.txt" );
        //SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	/*if( argc < 2 || argc > 2 ) {
		std::cerr << "Wrong argument count - must be in form: 'program file'\n";
		return 1;
	}

	std::string filename = argv[1];*/


	/*std::cout << "Loading file...";
	M4D::Imaging::AImage::Ptr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( filename );
	std::cout << "Done\n";
	

	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > prodconn;
	prodconn.PutDataset( image );*/


	QApplication app(argc, argv);
	try {
		processCommandLine( argc, argv );

		std::cout << "Show window\n";
		//ViewerWindow viewer( prodconn );
		ViewerWindow viewer;


		viewer.show();
		if ( !inFilename.empty() ) {
			viewer.openFile( QString::fromStdString( inFilename ) );
		}
		return app.exec();
	} catch ( std::exception &e )
	{
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
	} 
	catch (...) {
		QMessageBox::critical ( NULL, "Exception", "Unknown error" );
	}
	
	return 1;
}

