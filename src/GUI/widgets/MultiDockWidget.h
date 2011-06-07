#ifndef MULTI_DOCK_WIDGET_H
#define MULTI_DOCK_WIDGET_H

#include <QMoveEvent>
#include <QtGui>
#include <QtCore>
#include "common/Common.h"
#include "GUI/utils/QtM4DTools.h"

class TitleBarWidget: public QWidget
{
	Q_OBJECT;
public:
	TitleBarWidget( const QString &aTitle, QWidget * parent = 0 ): QWidget( parent )
	{
		resize( QSize( 100, 15 ) );
		setMinimumSize( QSize( 50, 12 ) );
		
		QHBoxLayout *layout = new QHBoxLayout;
		QLabel *label = new QLabel( aTitle );
		layout->addWidget( label );
		setLayout( layout );
	}
	QPoint
	getClickPos()
	{
		return mClickPos;
	}	
protected:
	void	
	mouseDoubleClickEvent ( QMouseEvent * event )
	{
		event->ignore();
	}
	
	void
	mouseMoveEvent ( QMouseEvent * event )
	{
		event->ignore();
	}
	
	void
	mousePressEvent ( QMouseEvent * event )
	{
		mClickPos = event->pos();
		LOG( "Mouse event - local: " << event->x() << "; " << event->y() );
		LOG( "Mouse event - global: " << event->globalX() << "; " << event->globalY() );
		event->ignore();
	}
	
	void
	mouseReleaseEvent ( QMouseEvent * event )
	{
		event->ignore();
	}
private:
	QPoint mClickPos;

};

class DockWidgetPrivate: public QDockWidget
{
	Q_OBJECT;
public:
	DockWidgetPrivate( const QString & aTitle ): QDockWidget( aTitle )
	{
		mTitleBar = new TitleBarWidget( aTitle, this );
		setTitleBarWidget( mTitleBar );
		//titleBar->show();
	}
signals:
	void
	dockMoved( QPoint globPos, QPoint locPos );
protected:
	void
	moveEvent ( QMoveEvent * event )
	{
		QDockWidget::moveEvent( event );
		emit dockMoved( event->pos(), mTitleBar->getClickPos() );
	}
	TitleBarWidget *mTitleBar;
};

class MultiDockWidget: public QWidget
{
	Q_OBJECT;
public:
	MultiDockWidget ( const QString & aTitle, QWidget * parent = 0 ):
		QWidget( parent ), mWidget(NULL ), mTitle( aTitle ), mCurrentMainWindow( NULL ), mCurrentDockWidget( NULL )
	{
		//QObject::connect( this, SIGNAL( continueInDrag( QWidget *, QPoint ) ), this, SLOT( continueInDragSlot( QWidget *, QPoint ) ), Qt::QueuedConnection );
		//mCurrentDockWidget = new DockWidgetPrivate( mTitle );
		//QObject::connect( mCurrentDockWidget, SIGNAL( dockMoved( QPoint, QPoint ) ), this, SLOT( dockMoved( QPoint, QPoint ) ), Qt::QueuedConnection );
		setLayout( new QVBoxLayout() );
	}
	
	void
	addDockingWindow( Qt::DockWidgetArea aArea, QMainWindow *aWin )
	{
		ASSERT( aWin );

		WinRecord rec;
		rec.area = aArea;
		rec.win = aWin;

		mDockingWindows.push_back( rec );

		if( mCurrentMainWindow == NULL ) {
			switchCurrentMainWindow( rec, QPoint(), QPoint(), false );
			//mCurrentMainWindow = aWin;
			//mCurrentMainWindow->addDockWidget( Qt::RightDockWidgetArea, mCurrentDockWidget );
		}
	}
	void
	setWidget( QWidget *aWidget )
	{
		ASSERT( aWidget );
		mWidget = aWidget;
		layout()->addWidget( aWidget );
	}
signals:
	/*void
	continueInDrag( QWidget *titleBar, QPoint pos );*/
protected:
	struct WinRecord
	{
		Qt::DockWidgetArea area;
		QMainWindow *win;
	};
	void
	switchCurrentMainWindow( WinRecord aRecord, QPoint globPos, QPoint locPos, bool dragging )
	{
		ASSERT( aRecord.win );
		
		LOG( "Switching main window" );

		/*mCurrentDockWidget->setVisible( false );
		mCurrentMainWindow->removeDockWidget( mCurrentDockWidget );
		mCurrentMainWindow = aRecord.win;
		mCurrentMainWindow->addDockWidget( Qt::RightDockWidgetArea, mCurrentDockWidget );
		mCurrentDockWidget->setVisible( true );*/
		DockWidgetPrivate *dock = new DockWidgetPrivate( mTitle );
	
		aRecord.win->addDockWidget( aRecord.area, dock );
		dock->setWidget( this );
		dock->setVisible( false );
		if( mCurrentDockWidget ) {
			dock->setFloating( mCurrentDockWidget->isFloating() );
			dock->resize( mCurrentDockWidget->size() );
			dock->move( mCurrentDockWidget->pos() );
			if (mCurrentMainWindow) {
				mCurrentMainWindow->removeDockWidget( mCurrentDockWidget );
			}
			delete mCurrentDockWidget;
		}
		mCurrentDockWidget = dock;
		mCurrentMainWindow = aRecord.win;
		dock->setVisible( true );
		QObject::connect( mCurrentDockWidget, SIGNAL( dockMoved( QPoint, QPoint ) ), this, SLOT( dockMoved( QPoint, QPoint ) ), Qt::QueuedConnection );
		if (dragging) {
			//emit continueInDrag( titleBar, pos );
			LOG( "sending event " << M4D::GUI::QPointToVector2i( locPos ) );
			QMouseEvent *event = new QMouseEvent( QEvent::MouseButtonPress, locPos, globPos, Qt::LeftButton, Qt::LeftButton, Qt::NoModifier );
			QCoreApplication::postEvent ( dock->titleBarWidget(), event );
			//QCoreApplication *app = QCoreApplication::instance();
			//app->notify( titleBar, &event );
		}
	}
protected slots:
	void
	dockMoved( QPoint globPos, QPoint locPos )
	{
		LOG( "Pos :" << globPos.x() << "; " << globPos.y() );
		LOG( "LPos :" << locPos.x() << "; " << locPos.y() );
		if( globPos.x() == 1000 ) {
			switchCurrentMainWindow( mDockingWindows[1], globPos, locPos, true );
		}
	}

	/*void
	continueInDragSlot( QWidget *titleBar, QPoint pos )
	{
		LOG( "sending event " << M4D::GUI::QPointToVector2i( titleBar->mapToGlobal( titleBar->pos() ) ) );
		QMouseEvent *event = new QMouseEvent( QEvent::MouseButtonPress, titleBar->mapFromGlobal( pos ), pos, Qt::LeftButton, Qt::LeftButton, Qt::NoModifier );
		QCoreApplication::postEvent ( titleBar, event );
			//QCoreApplication *app = QCoreApplication::instance();
			//app->notify( titleBar, &event );
	}*/
private:
	QWidget *mWidget;
	QString mTitle;
	QList< WinRecord > mDockingWindows;
	QMainWindow *mCurrentMainWindow;
	DockWidgetPrivate *mCurrentDockWidget;
};


#endif /*MULTI_DOCK_WIDGET_H*/
