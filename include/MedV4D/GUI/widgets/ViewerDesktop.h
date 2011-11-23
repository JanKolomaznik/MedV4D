#ifndef VIEWER_DESKTOP_H
#define VIEWER_DESKTOP_H

#include <QtGui>
#include "GUI/widgets/AGLViewer.h"
#include <boost/bind/bind.hpp>
#include "MedV4D/Imaging/ConnectionInterface.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{

class AViewerFactory
{
public:
	typedef boost::shared_ptr< AViewerFactory > Ptr;
	virtual AGLViewer *
	createViewer() = 0;
};

class ViewerDesktop: public QWidget
{
	Q_OBJECT;
public:
	ViewerDesktop( QWidget *parent = NULL );

	template< typename TFunctor >
	TFunctor
	forEachViewer( TFunctor ftor );

	/*void
	setInputConnection( M4D::Imaging::ConnectionInterface &mProdconn )
	{
		for ( ViewerList::iterator it = mViewers.begin(); it != mViewers.end(); ++it ) {
			GeneralViewer * viewer = dynamic_cast< GeneralViewer * >( it->viewer );
			if ( viewer ) {
				mProdconn.ConnectConsumer( viewer->InputPort()[0] );
			}
		}
	}*/
	void
	setViewerFactory( AViewerFactory::Ptr aFactory )
	{
		mViewerFactory = aFactory;
	}
public slots:
	void
	setLayoutOrganization( int cols, int rows );

	void
	updateAllViewers()
	{
		for ( ViewerList::iterator it = mViewers.begin(); it != mViewers.end(); ++it ) {
			it->viewer->update();
		}
	}

signals:
	void
	updateInfo( const QString & aInfo );

protected:
	AGLViewer *
	createViewer();
	
	struct ViewerInfo
	{
		AGLViewer *viewer;
	};

	typedef std::vector< ViewerInfo > ViewerList;

	ViewerList mViewers;

	AViewerFactory::Ptr mViewerFactory;

};

template< typename TFunctor >
TFunctor
ViewerDesktop::forEachViewer( TFunctor ftor )
{
	for ( ViewerList::iterator it = mViewers.begin(); it != mViewers.end(); ++it ) {
		ftor( it->viewer );
	}
	return ftor;
}


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#endif /*VIEWER_DESKTOP_H*/
