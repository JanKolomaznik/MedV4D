#ifndef QT_MODEL_VIEW_TOOLS_H
#define QT_MODEL_VIEW_TOOLS_H

//Temporary workaround
#ifndef Q_MOC_RUN
#include <QtWidgets>

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/GeometricPrimitives.h"
#include "MedV4D/Common/Sphere.h"
#endif

class AVectorItemModel: public QAbstractListModel
{
public:
	AVectorItemModel( QObject *parent = 0): QAbstractListModel(parent)
	{}

	virtual void clear()=0;
};

template< typename TType >
struct TypeQtMVTraits;

template< typename TCoordType >
struct TypeQtMVTraits< M4D::Point3D< TCoordType > >
{
	typedef M4D::Point3D< TCoordType > Type;
	static const int colCount = 3;
	static int
	columnCount(){ return colCount; }

	static QVariant
	headerData( int section )
	{
		switch( section ) {
		case 0:
			return QString("X");
		case 1:
			return QString("Y");
		case 2:
			return QString("Z");
		default:
			return QVariant();
		}
	}
	static QVariant
	data( int column, const Type &item )
	{
		ASSERT( column >= 0 && column < colCount );
		return item[ column ];
	}
};

template< unsigned taDim, typename TCoordType >
struct TypeQtMVTraits< M4D::Sphere< taDim, TCoordType > >
{
	typedef M4D::Sphere< taDim, TCoordType > Type;
	static const int colCount = taDim + 1;
	static int
	columnCount(){ return colCount; }

	static QVariant
	headerData( int section )
	{
		if ( section > (int)taDim ) {
			return QVariant();
		}
		switch( section ) {
		case 0:
			return QString("Radius");
		case 1:
			return QString("X");
		case 2:
			return QString("Y");
		case 3:
			return QString("Z");
		default:
			return QVariant();
		}
	}
	static QVariant
	data( int column, const Type &item )
	{
		ASSERT( column >= 0 && column < colCount );
		if( column == 0 ) {
			return item.radius();
		}
		return item.center()[ column-1 ];
	}
};

template< typename TType, unsigned tDim >
struct TypeQtMVTraits< M4D::Line< TType, tDim > >
{
	typedef M4D::Line< TType, tDim > Type;
	static const int colCount = 2*tDim;
	static int
	columnCount(){ return colCount; }

	static QVariant
	headerData( int section )
	{
		if ( section > (int)(2*tDim) ) {
			return QVariant();
		}
		if (section < (int)tDim ) {
			switch( section ) {
			case 0:
				return QString("X1");
			case 1:
				return QString("Y1");
			case 2:
				return QString("Z1");
			default:
				return QVariant();
			}
		} else {
			switch( section - tDim) {
			case 0:
				return QString("X2");
			case 1:
				return QString("Y2");
			case 2:
				return QString("Z2");
			default:
				return QVariant();
			}
		}

	}
	static QVariant
	data( int column, const Type &item )
	{
		ASSERT( column >= 0 && column < colCount );
		if( column < (int)tDim ) {
			return item.firstPoint()[column];
		}
		return item.secondPoint()[column-tDim];
	}
};
//**************************************************************************

template< typename TType, template < typename > class TTraits = TypeQtMVTraits >
class VectorItemModel:public AVectorItemModel, public std::vector< TType >
{
public:
	typedef std::vector< TType > Container;
	typedef TTraits< TType > Traits;

	VectorItemModel( QObject *parent = 0): AVectorItemModel(parent)
	{}

	int
	rowCount(const QModelIndex &parent = QModelIndex()) const
	{
		return this->size();
	}

	void
	push_back( const typename Container::value_type &aVal )
	{
		QAbstractListModel::beginInsertRows(QModelIndex(), this->size(), this->size() );
		Container::push_back( aVal );
		QAbstractListModel::endInsertRows();
	}

	void
	clear()
	{
		QAbstractListModel::beginRemoveRows(QModelIndex(), 0, this->size()-1 );
		Container::clear();
		QAbstractListModel::endRemoveRows();
	}

	int
	columnCount(const QModelIndex &parent = QModelIndex()) const
	{ return Traits::columnCount(); }

	QVariant
	data(const QModelIndex &index, int role) const
	{
		if (!index.isValid())
			return QVariant();

		if ( index.row() >= (int)this->size() || index.column() >= Traits::columnCount() )
			return QVariant();

		if (role == Qt::DisplayRole) {
			return Traits::data( index.column(), this->at( index.row() ) );
		}

		return QVariant();
	}

	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const
	{
		if (role != Qt::DisplayRole)
			return QVariant();

		if( orientation == Qt::Vertical ) {
			return section;
		}
		return Traits::headerData( section );
	}
};

/*class PointSet : public VectorItemModel< M4D::Point3Df >
{
	Q_OBJECT;
public:

	PointSet( QObject *parent = 0): VectorItemModel< M4D::Point3Df >(parent)
	{}

	int
	columnCount(const QModelIndex &parent = QModelIndex()) const
	{return 3;}

	QVariant
	data(const QModelIndex &index, int role) const
	{
		if (!index.isValid())
			return QVariant();

		if ( index.row() >= (int)size() || index.column() > 2 )
			return QVariant();

		if (role == Qt::DisplayRole) {
			return at( index.row() )[index.column()];
		}

		return QVariant();
	}

	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const
	{
		if (role != Qt::DisplayRole)
			return QVariant();

		if( orientation == Qt::Vertical ) {
			return section;
		}
		switch( section ) {
		case 0:
			return QString("X");
		case 1:
			return QString("Y");
		case 2:
			return QString("Z");
		default:
			return QVariant();
		}
		return QVariant();
	}
};

class SphereSet : public VectorItemModel< M4D::Sphere3Df >
{
	Q_OBJECT;
public:
	SphereSet( QObject *parent = 0): VectorItemModel< M4D::Sphere3Df >(parent)
	{

	}

	int
	columnCount(const QModelIndex &parent = QModelIndex()) const
	{
		return 4;
	}

	QVariant
	data(const QModelIndex &index, int role) const
	{
		if (!index.isValid())
			return QVariant();

		if ( index.row() >= (int)size() || index.column() > 3 )
			return QVariant();

		if (role == Qt::DisplayRole) {
			if( index.column() <= 2 ) {
				return at( index.row() ).center()[index.column()];
			} else {
				return at( index.row() ).radius();
			}
		}

		return QVariant();
	}

	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const
	{
		if (role != Qt::DisplayRole)
			return QVariant();

		if( orientation == Qt::Vertical ) {
			return section;
		}
		switch( section ) {
		case 0:
			return QString("X");
		case 1:
			return QString("Y");
		case 2:
			return QString("Z");
		case 3:
			return QString("Radius");
		default:
			return QVariant();
		}
		return QVariant();
	}
};

class LineSet : public VectorItemModel<  M4D::Line3Df >
{
	Q_OBJECT;
public:
	LineSet( QObject *parent = 0): VectorItemModel<  M4D::Line3Df >(parent)
	{

	}

	int
	columnCount(const QModelIndex &parent = QModelIndex()) const
	{
		return 6;
	}

	QVariant
	data(const QModelIndex &index, int role) const
	{
		if (!index.isValid())
			return QVariant();

		if ( index.row() >= (int)size() || index.column() > 5 )
			return QVariant();

		if (role == Qt::DisplayRole) {
			if ( index.column() < 3 ) {
				return at( index.row() ).firstPoint()[index.column()];
			} else {
				return at( index.row() ).secondPoint()[index.column()-3];
			}
		}

		return QVariant();
	}

	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const
	{
		if (role != Qt::DisplayRole)
			return QVariant();

		if( orientation == Qt::Vertical ) {
			return section;
		}
		switch( section ) {
		case 0:
			return QString("X1");
		case 1:
			return QString("Y1");
		case 2:
			return QString("Z1");
		case 3:
			return QString("X2");
		case 4:
			return QString("Y2");
		case 5:
			return QString("Z2");
		default:
			return QVariant();
		}
		return QVariant();
	}
};*/


#endif /*QT_MODEL_VIEW_TOOLS_H*/
