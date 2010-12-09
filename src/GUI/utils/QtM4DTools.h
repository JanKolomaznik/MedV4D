#ifndef QT_M4D_TOOLS_H
#define QT_M4D_TOOLS_H

#include "common/Common.h"
#include <QPointF>
#include <QPoint>
#include <QSize>
#include <QString>

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

}//namespace GUI
}//namespace M4D

#endif /*QT_M4D_TOOLS_H*/
