#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
//#include "MedV4D/GUI/widgets/m4dGUISliceViewerWidget.h"
#include "MedV4D/GUI/widgets/BasicSliceViewer.h"
#include "Imaging/Imaging.h"
#include "MedV4D/Common/Common.h"


class ViewerWindow : public QWidget
{
private:
	M4D::GUI::Viewer::BasicSliceViewer *viewerWidget;
public:
	ViewerWindow( M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > & conn);
	//ViewerWindow( M4D::Imaging::AImage::Ptr image );
	~ViewerWindow();

	void
	SetLUTWindow( Vector< float32, 2 > window );
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

}

void ViewerWindow::SetLUTWindow( Vector< float32, 2 > window) 
{
	viewerWidget->SetLUTWindow(window);	
}
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

ViewerWindow::~ViewerWindow()
{}

int
main( int argc, char** argv )
{
	//std::ofstream logFile( "Log.txt" );
        //SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	if( argc < 2 || argc > 2 ) {
		std::cerr << "Wrong argument count - must be in form: 'program file'\n";
		return 1;
	}

	std::string filename = argv[1];


	std::cout << "Loading file...";
	M4D::Imaging::AImage::Ptr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( filename );
	std::cout << "Done\n";
	

	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > prodconn;
	prodconn.PutDataset( image );


	QApplication app(argc, argv);
	try {
		std::cout << "Show window\n";
		ViewerWindow viewer( prodconn );
		//ViewerWindow viewer( image );
		viewer.show();

		Vector<float32, 2> wl;
		wl[0] = (350.0f ) / 65536.0f ;
		wl[1] = (40 + 1000.0f) / 65536.0f ;
		viewer.SetLUTWindow(wl);

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

