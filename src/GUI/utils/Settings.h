#ifndef SETTINGS_H
#define SETTINGS_H

#include "common/Common.h"
#include "boost/variant.hpp"
#include <map>
#include <iterator>

class Settings: public QAbstractListModel
{
public:
	typedef boost::variant< bool, int, double, std::string > ValueVariant;
	struct ValueRecord
	{
		ValueVariant value;
		std::string name;
	};
	void
	loadFromXML( boost::filesystem::path aPath );

	void
	saveToXML( boost::filesystem::path aPath );

	template< typename TType >
	TType
	get( std::string aName, TType aDefaultValue );

	template< typename TType >
	void
	set( std::string aName, TType aValue );


	int 
	rowCount(const QModelIndex &parent = QModelIndex()) const
	{
		return mValues.size();
	}

	int 
	columnCount(const QModelIndex &parent = QModelIndex()) const
	{
		return 3;
	}

	QVariant 
	data(const QModelIndex &index, int role) const
	{
		if (!index.isValid())
			return QVariant();

		if ( index.row() >= (int)mValues.size() || index.column() > 2 )
			return QVariant();
		
		if (role == Qt::DisplayRole) {
			ValueMap::const_iterator it = mValues.begin();
			std::advance( it, index.row() );
			const ValueRecord &rec = it->second;
			switch( index.column() ) {
			case 0:
				return QString( rec.name.data() );
			case 1:
				return QVariant();
			case 2:
				return QVariant();
			default:
				return QVariant();
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
			return QVariant();
		}
		switch( section ) {
		case 0:
			return QString(tr("Option name"));
		case 1:
			return QString(tr("Value"));
		case 2:
			return QString(tr("Type"));
		default:
			return QVariant();
		}
		return QVariant();
	}
protected:
	typedef std::map< std::string, ValueRecord > ValueMap;
	ValueMap mValues;
};

template< typename TType >
TType
Settings::get( std::string aName, TType aDefaultValue )
{
	ASSERT( ! aName.empty() );

	ValueMap::iterator it = mValues.find(aName);
	if ( it != mValues.end() ) {
		return boost::get< TType >( it->second.value );
	}

	ValueRecord rec;
	rec.value = aDefaultValue;
	rec.name = aName;
	mValues[ aName ] = rec;
	return aDefaultValue;
}

template< typename TType >
void
Settings::set( std::string aName, TType aValue )
{
	ASSERT( ! aName.empty() );
	ValueRecord rec;
	rec.value = aValue;
	rec.name = aName;
	mValues[ aName ] = rec;
}



#endif /*SETTINGS_H*/
