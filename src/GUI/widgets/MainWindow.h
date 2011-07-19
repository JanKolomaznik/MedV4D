#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtGui>
#include <boost/shared_ptr.hpp>

#include "GUI/widgets/MultiDockWidget.h"

namespace M4D
{
namespace GUI
{

class MainWindow: public QMainWindow
{
	Q_OBJECT;
public:
	MainWindow();

	void
	createDockWidget( QString aName, Qt::DockWidgetArea aArea, QWidget * aWidget )
	{
		if ( aWidget == NULL ) {
			_THROW_ M4D::ErrorHandling::ENULLPointer();
		}
		MultiDockWidget *dockwidget = new MultiDockWidget( aName );
		dockwidget->setWidget( aWidget );
		dockwidget->addDockingWindow( aArea, this );
	}
protected:

private:

};


} /*namespace GUI*/
} /*namespace M4D*/


#endif /*MAIN_WINDOW_H*/


