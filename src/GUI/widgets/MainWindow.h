#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtGui>
#include <boost/shared_ptr.hpp>
#include "tmp/ui_MainWindow.h"

namespace M4D
{
namespace GUI
{

class MainWindow: public QMainWindow, public Ui::MainWindow
{
	Q_OBJECT;
public:
	MainWindow();

	virtual void
	Initialize()
	{}

protected:

private:

};


} /*namespace GUI*/
} /*namespace M4D*/


#endif /*MAIN_WINDOW_H*/


