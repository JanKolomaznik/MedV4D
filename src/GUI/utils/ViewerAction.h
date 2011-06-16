#ifndef VIEWER_ACTION_H
#define VIEWER_ACTION_H

#include <QtGui>
#include "GUI/widgets/GeneralViewer.h"
#include "GUI/utils/ViewerManager.h"
#include "GUI/utils/ApplicationManager.h"
//#include <boost/cast.hpp>
#include "common/Functors.h"

class HelperViewerActionInterface
{
public:
	virtual void
	callCheckState() = 0;
};

class ViewerActionSet: public QObject
{
	Q_OBJECT;
public:
	ViewerActionSet()
	{
		QObject::connect( ApplicationManager::getInstance(), SIGNAL( selectedViewerSettingsChanged() ), this, SLOT( updateActions() ) );
	}

	~ViewerActionSet();

	void
	addAction( QAction * aAction )
	{
		ASSERT( aAction );
		mActions.push_back( aAction );
		aAction->setParent( this );
		//QObject::connect( aAction, SIGNAL( triggered( bool ) ), this, SLOT( updateActions() ) );
		
	}

	void
	addActionGroup( QActionGroup * aActionGroup )
	{
		ASSERT( aActionGroup );
		mActionGroups.push_back( aActionGroup );
		aActionGroup->setParent( this );
		mActions.append( aActionGroup->actions() );
		//QObject::connect( aActionGroup, SIGNAL( triggered(QAction *) ), this, SLOT( updateActions() ) );
	}
	
	QList<QAction *> &
	getActions()
	{
		return mActions;
	}
	
	void
	addSeparator()
	{
		QAction *sep = new QAction( this );
		sep->setSeparator( true );
		mActions.push_back( sep );
	}
public slots:
	void
	updateActions()
	{
		QList<QAction *>::iterator it;
		for ( it = mActions.begin(); it != mActions.end(); ++it ) {
			HelperViewerActionInterface *action = dynamic_cast< HelperViewerActionInterface* >( *it );
			if ( action ) {
				action->callCheckState();
			}
		}
	}

protected:

	QList<QAction *> mActions;
	QList<QActionGroup *> mActionGroups;
};

QToolBar *
createToolBarFromViewerActionSet( ViewerActionSet &aActionSet, const QString &aName );

template < typename TWidget >
void
addViewerActionSetToWidget( TWidget &aWidget, ViewerActionSet &aActionSet )
{
	QList<QAction *> &actions = aActionSet.getActions();
	QList<QAction *>::iterator it;
	for ( it = actions.begin(); it != actions.end(); ++it ) {
		aWidget.addAction( *it );
	}
}

class HelperViewerAction: public QAction, public HelperViewerActionInterface
{
	Q_OBJECT;
public:
	HelperViewerAction( QString aName, QObject *parent );

public slots:
	void
	callCheckState(){};
protected slots:
	virtual void
	callAction()=0;
protected:
};


template< typename TViewer, typename TOnClick, typename TCheckState, typename TCheckEnabled >
class ViewerAction: public HelperViewerAction
{
public:
	ViewerAction( QString aName, TOnClick aOnClick, TCheckState aCheckState, TCheckEnabled aCheckEnabled, QObject *parent )
		: HelperViewerAction( aName, parent ), mOnClick( aOnClick ), mCheckState( aCheckState ), mCheckEnabled( aCheckEnabled )
	{
		//setCheckable( tCheckable );		
		QObject::connect( ApplicationManager::getInstance(), SIGNAL( viewerSelectionChanged() ), this, SLOT( callCheckState() ) );
	}

	void
	callCheckState()
	{
		TViewer *viewer = getSelectedViewer();
		if( viewer == NULL ) {
			setEnabled( false );
		} else {
			bool enable = mCheckEnabled( viewer );
			setEnabled( enable );
			if ( isCheckable() ) {
				setChecked( mCheckState( viewer ) );
			}
		}
	}

protected:
	TViewer *
	getSelectedViewer()
	{
		return dynamic_cast< TViewer * >( ViewerManager::getInstance()->getSelectedViewer() );
	}

	void
	callAction()
	{
		TViewer *viewer = getSelectedViewer();
		if( viewer == NULL ) {
			return;
		}

		mOnClick( viewer, isChecked() );
	}
	TOnClick mOnClick;
	TCheckState mCheckState;
	TCheckEnabled mCheckEnabled;
};

template< typename TOnClick, typename TCheckState, typename TCheckEnabled >
QAction *
createGeneralViewerAction( QString aName, TOnClick aOnClick, TCheckState aCheckState, TCheckEnabled aCheckEnabled, bool checkable, QObject *parent = NULL )
{
	HelperViewerAction *action = new ViewerAction< M4D::GUI::Viewer::GeneralViewer, TOnClick, TCheckState, TCheckEnabled >( aName, aOnClick, aCheckState, aCheckEnabled, parent );
	action->setCheckable( checkable );
	action->callCheckState();
	return action;
}

template< typename TOnClick, typename TCheckState >
QAction *
createGeneralViewerAction( QString aName, TOnClick aOnClick, TCheckState aCheckState, bool checkable, QObject *parent = NULL)
{
	return createGeneralViewerAction( aName, aOnClick, aCheckState, M4D::Functors::PredicateAlwaysTrue(), checkable, parent );
}
//**************************************************************
class HelperViewerWidgetAction: public QWidgetAction, public HelperViewerActionInterface
{
	Q_OBJECT;
public:
	HelperViewerWidgetAction( QObject *parent ): QWidgetAction( parent )
	{}

public slots:
	void
	callCheckState(){}

	virtual void
	callChangeOption( const QString & aOption ) = 0;
};

template< typename TViewer, typename TOnChangeOption, typename TGetOptions, typename TGetCurrentOption, typename TCheckEnabled >
class SubmenuViewerAction: public HelperViewerWidgetAction
{
public:
	SubmenuViewerAction( QString aName, TOnChangeOption aOnChangeOption, TGetOptions aGetOptions, TGetCurrentOption aGetCurrentOption, TCheckEnabled aCheckEnabled, QObject *parent )
		: HelperViewerWidgetAction( parent ), mName( aName ), mOnChangeOption( aOnChangeOption ), mGetOptions( aGetOptions ), mGetCurrentOption( aGetCurrentOption ), mCheckEnabled( aCheckEnabled )
	{
		QObject::connect( ApplicationManager::getInstance(), SIGNAL( viewerSelectionChanged() ), this, SLOT( callCheckState() ) );
	}

	void
	callCheckState()
	{
		/*static int counter = 0;
		++counter;
		D_BLOCK_COMMENT( TO_STRING( "callCheckState start - " << counter ), TO_STRING( "callCheckState end - " << counter ) );*/

		TViewer *viewer = getSelectedViewer();
		if( viewer == NULL ) {
			setEnabled( false );
			QList<QWidget *>::iterator it;
			QList<QWidget *> widgets = createdWidgets();
			for ( it = widgets.begin(); it != widgets.end(); ++it ) {
				(*it)->setEnabled( false );
			}
		} else {
			bool enable = mCheckEnabled( viewer );
			setEnabled( enable );
			QList<QWidget *>::iterator it;
			QList<QWidget *> widgets = createdWidgets();
			for ( it = widgets.begin(); it != widgets.end(); ++it ) {
				(*it)->setEnabled( enable );
			}

			mStrings = mGetOptions( viewer );
			mCurrentOption = mGetCurrentOption( viewer );
			rebuildWidgets( mStrings, mCurrentOption );
		}
	}

	void
	callChangeOption( const QString & aOption )
	{
		/*static int counter = 0;
		++counter;
		D_BLOCK_COMMENT( TO_STRING( "callChangeOption start - " << counter ), TO_STRING( "callChangeOption end - " << counter ) );*/

		TViewer *viewer = getSelectedViewer();
		if( viewer == NULL ) {
			return;
		}
		mCurrentOption = aOption;
		mOnChangeOption( viewer, aOption );
		//actualizeWidgets( mOption );
	}
protected:
	TViewer *
	getSelectedViewer()
	{
		return dynamic_cast< TViewer * >( ViewerManager::getInstance()->getSelectedViewer() );
	}

	QWidget *
	createWidget ( QWidget * parent )
	{
		if ( parent->inherits( "QMenu" ) ) {
			/*QMenu *menu = new QMenu( "Color transform" );
			QAction * testAction = new QAction( "1111", menu );
			menu->addAction( testAction );
			testAction = new QAction( "2222", menu );
			menu->addAction( testAction );

			dynamic_cast< QMenu* >( parent )->addMenu( menu );*/
			
			return NULL;
		}
		if ( parent->inherits( "QToolBar" ) ) {
			QComboBox *combo = new QComboBox( parent );
			combo->setSizeAdjustPolicy( QComboBox::AdjustToContents );
			combo->setEnabled( isEnabled() );
			rebuildCombo( mStrings, mCurrentOption, combo );
			QObject::connect( combo, SIGNAL( currentIndexChanged( const QString & ) ), this, SLOT( callChangeOption( const QString & ) ) ); 
			return combo;
		}
		D_PRINT( "Unhandled parent class: " << parent->metaObject()->className() );
		return NULL;
	}

	//QList<QWidget *>	createdWidgets () const
	void
	deleteWidget ( QWidget * widget )
	{
		widget->deleteLater();
	}

	void
	actualizeWidgets( QString aCurrent )
	{
		//actualizeActionGroup( aCurrent );

		//actualizeMenu( NULL );

		//actualizeCombo( aCurrent, NULL );
	}

	void
	rebuildWidgets( QStringList &aStrings, QString aCurrent )
	{
		rebuildActionGroup( aStrings, aCurrent );

		rebuildMenu( NULL );

		QList<QWidget *>::iterator it;
		QList<QWidget *> widgets = createdWidgets();
		for ( it = widgets.begin(); it != widgets.end(); ++it ) {
			QComboBox *combo = dynamic_cast< QComboBox *>( *it );
			if ( combo ) {
				rebuildCombo( aStrings, aCurrent, combo );
			}
		}
	}

	void
	rebuildActionGroup( QStringList &aStrings, QString aCurrent )
	{}

	void
	rebuildMenu( QMenu *aMenu )
	{}

	void
	rebuildCombo( QStringList &aStrings, QString aCurrent, QComboBox *aComboBox )
	{
		ASSERT( aComboBox );
		int idx = -1;
		for (int i = 0; i < aStrings.size(); ++i ) {
			if ( aStrings[ i ] == aCurrent ) {
				idx = i;
				break;
			}
		}
		aComboBox->blockSignals( true );
		aComboBox->clear();
		aComboBox->addItems(aStrings);
		aComboBox->setCurrentIndex( idx );
		aComboBox->blockSignals( false );
	}


	QString mName;
	QList< QMenu * > mMenus;
	QActionGroup *mActionGroup;
	QStringList mStrings;
	QString mCurrentOption;

	TOnChangeOption mOnChangeOption;
	TGetOptions mGetOptions;
	TGetCurrentOption mGetCurrentOption;
	TCheckEnabled mCheckEnabled;
};

template< typename TOnChangeOption, typename TGetOptions, typename TGetCurrentOption, typename TCheckEnabled >
QAction *
createGeneralViewerSubmenuAction( QString aName, TOnChangeOption aOnChangeOption, TGetOptions aGetOptions, TGetCurrentOption aGetCurrentOption, TCheckEnabled aCheckEnabled, QObject *parent = NULL )
{
	QAction *action = new SubmenuViewerAction< M4D::GUI::Viewer::GeneralViewer, TOnChangeOption, TGetOptions, TGetCurrentOption, TCheckEnabled >( aName, aOnChangeOption, aGetOptions, aGetCurrentOption, aCheckEnabled, parent );
	return action;
}



#endif /*VIEWER_ACTION_H*/
