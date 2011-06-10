#ifndef VIEWER_DESKTOP_H
#define VIEWER_DESKTOP_H

#include <QtGui>
#include "GUI/widgets/AGLViewer.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{



class ViewerDesktop: public QWidget
{
	Q_OBJECT;
public:
	ViewerDesktop( QWidget *parent = NULL );

	template< typename TFunctor >
	TFunctor
	forEachViewer( TFunctor ftor );

public slots:
	void
	setLayoutOrganization( int cols, int rows );

protected:
	AGLViewer *
	createViewer();
	
	struct ViewerInfo
	{
		AGLViewer *viewer;
	};

	typedef std::vector< ViewerInfo > ViewerList;

	ViewerList mViewers;

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
