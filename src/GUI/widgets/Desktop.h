#ifndef VIEWER_DESKTOP_H
#define VIEWER_DESKTOP_H

#include <QtGui>
#include <boost/shared_ptr.hpp>
#include "GUI/widgets/ADesktop.h"
#include "GUI/widgets/ADesktopContainer.h"

namespace M4D
{
namespace GUI
{


class Desktop: public QWidget
{
	Q_OBJECT;
public:
	Desktop( QWidget *parent = NULL );

	~Desktop();

protected:
	std::vector< ADesktopContainer * >	_desktopContainers;
private:

};



} /*namespace GUI*/
} /*namespace M4D*/


#endif /*VIEWER_DESKTOP_H*/

