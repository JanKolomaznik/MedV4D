#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtGui>
#include <boost/shared_ptr.hpp>

#include "MedV4D/GUI/widgets/MultiDockWidget.h"
#include "MedV4D/GUI/widgets/AGLViewer.h"
#include "MedV4D/GUI/widgets/GeneralViewer.h"

namespace M4D
{
namespace GUI
{

class MainWindow: public QMainWindow
{
	Q_OBJECT;
public:
	MainWindow();

	MultiDockWidget *
	createDockWidget( QString aName, Qt::DockWidgetArea aArea, QWidget * aWidget, bool aShow = true )
	{
		if ( aWidget == NULL ) {
			_THROW_ M4D::ErrorHandling::ENULLPointer();
		}
		MultiDockWidget *dockwidget = new MultiDockWidget( aName, this );
		dockwidget->setWidget( aWidget );
		addDockWidget( aArea, dockwidget );
		//dockwidget->addDockingWindow( aArea, this );
		dockwidget->setVisible(aShow);
		return dockwidget;
	}

	virtual void
	addRenderingExtension( M4D::GUI::Viewer::RenderingExtension::Ptr aRenderingExtension ){}

	virtual void
	setViewerController( M4D::GUI::Viewer::AViewerController::Ptr aViewerController ){}
public slots:
	virtual void
	updateGui()
	{ update(); };
protected:

private:

};


} /*namespace GUI*/
} /*namespace M4D*/


#endif /*MAIN_WINDOW_H*/


