#ifndef VIEWER_WINDOW
#define VIEWER_WINDOW

#include <QtGui>
#include <QtCore>
#include "ui_ViewerWindow.h"

class ViewerWindow: public QMainWindow, public Ui::ViewerWindow
{
	Q_OBJECT;
public:
	ViewerWindow();

	~ViewerWindow();


public slots:
	void
	openFile();

	void 
	openFile( const QString aPath );
protected:
	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > mProdconn;

private:

};


#endif /*VIEWER_WINDOW*/
