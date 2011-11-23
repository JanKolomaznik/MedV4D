#include "GUI/utils/ViewerAction.h"


ViewerActionSet::~ViewerActionSet()
{
	/*QList<QAction *>::iterator it;
	for ( it = mActions.begin(); it != mActions.end(); ++it ) {
		delete *it;
	}
	QList<QActionGroup *>::iterator it2;
	for ( it2 = mActionGroups.begin(); it2 != mActionGroups.end(); ++it2 ) {
		delete *it2;
	}*/
}

QToolBar *
createToolBarFromViewerActionSet( ViewerActionSet &aActionSet, const QString &aName )
{
	QToolBar * toolbar = new QToolBar( aName );
	
	addViewerActionSetToWidget( *toolbar, aActionSet );
	return toolbar;
}



HelperViewerAction::HelperViewerAction( const QString & text, QObject *parent ): QAction( text, parent )
{
	QObject::connect( this, SIGNAL( triggered( bool ) ), this, SLOT( callAction() ) );
}

HelperViewerAction::HelperViewerAction ( const QIcon & icon, const QString & text, QObject * parent ): QAction( icon, text, parent )
{
	QObject::connect( this, SIGNAL( triggered( bool ) ), this, SLOT( callAction() ) );
}

