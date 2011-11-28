
#include "MedV4D/GUI/utils/QtM4DTools.h"

namespace M4D {
namespace GUI {


QToolBar *
createToolbarFromActions( const QString &aName, QList<QAction *> &actions )
{
	QToolBar *toolbar = new QToolBar( aName );
	addActionsToWidget( *toolbar, actions );
	return toolbar;
}

}//namespace GUI
}//namespace M4D
