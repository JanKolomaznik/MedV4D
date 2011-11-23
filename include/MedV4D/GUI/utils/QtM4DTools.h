#ifndef QT_M4D_TOOLS_H
#define QT_M4D_TOOLS_H

#include "common/Common.h"
#include <QPointF>
#include <QPoint>
#include <QSize>
#include <QString>
#include <QtGui>
#include <QtCore>

namespace M4D {
namespace GUI {

inline Vector2f
QPointFToVector2f( const QPointF &aPoint )
{
	return Vector2f( aPoint.x(), aPoint.y() );
}

inline Vector2i
QPointToVector2i( const QPoint &aPoint )
{
	return Vector2i( aPoint.x(), aPoint.y() );
}

inline Vector2u
QSizeToVector2u( const QSize &aSize )
{
	return Vector2u( aSize.width(), aSize.height() );
}

inline Vector2i
QSizeToVector2i( const QSize &aSize )
{
	return Vector2i( aSize.width(), aSize.height() );
}

template< typename TVector >
QString
VectorToQString( const TVector & aVector )
{
	QString res = "[ ";
	for( unsigned i = 0; i < TVector::Dimension - 1; ++i ) {
		res += QString::number( aVector[i] );
		res += "; ";
	}
	res += QString::number( aVector[TVector::Dimension - 1] );
	res += " ]";
	return res;
}


template< typename TType >
void
synchronizeBoolAndChecked( TType &aObject, bool &aVariable, bool aFrom )
{
	if( aFrom ) {
		aVariable = aObject.isChecked();
	} else {
		aObject.setChecked( aVariable );
	}
}

template< typename TType >
void
synchronizeBoolAndEnabled( TType &aObject, bool &aVariable, bool aFrom )
{
	if( aFrom ) {
		aVariable = aObject.isEnabled();
	} else {
		aObject.setEnabled( aVariable );
	}
}

template < typename TWidget >
void
addActionsToWidget( TWidget &aWidget, QList<QAction *> &actions )
{
	QList<QAction *>::iterator it;
	for ( it = actions.begin(); it != actions.end(); ++it ) {
		aWidget.addAction( *it );
	}
}

QToolBar *
createToolbarFromActions( const QString &aName, QList<QAction *> &actions );


}//namespace GUI
}//namespace M4D

#endif /*QT_M4D_TOOLS_H*/
