#ifndef GRID_DESKTOP_H
#define GRID_DESKTOP_H

#include <QtGui>
#include <boost/shared_ptr.hpp>
#include "GUI/widgets/Desktop.h"

namespace M4D
{
namespace GUI
{


class GridDesktop: public Desktop
{
	Q_OBJECT;
public:
	GridDesktop( QWidget *parent = NULL );

	~GridDesktop();



public slots:
	/** 
	 * Slot for changing the Desktop's layout - should be connected to Screen Layout Widget.
	 *
	 * @param rows number of rows in the new layout
	 * @param columns number of columns in the new layout
	 **/
	virtual void 
	SetDesktopLayout ( unsigned rows, unsigned columns );

protected:

private:

};



} /*namespace GUI*/
} /*namespace M4D*/


#endif /*GRID_DESKTOP_H*/

