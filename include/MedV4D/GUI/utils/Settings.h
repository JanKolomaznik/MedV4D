#ifndef SETTINGS_H
#define SETTINGS_H

#include "MedV4D/Common/Common.h"
#include "boost/variant.hpp"
#include <map>
#include <iterator>

class Settings: public QAbstractListModel
{
public:
	typedef boost::variant< bool, int, double, std::string, Vector3d, Vector4d > ValueVariant;
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
		return static_cast<int>(mValues.size());
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
				return boost::apply_visitor( ConvertToQVariantVisitor(), rec.value );
			case 2:
				return boost::apply_visitor( TypeNameVisitor(), rec.value );
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
	class ConvertToQVariantVisitor: public boost::static_visitor<QVariant>
	{
	public:
		QVariant
		operator()( const bool & aVal ) const
		{
			return QVariant( aVal );
		}

		QVariant
		operator()( const int & aVal ) const
		{
			return QVariant( aVal );
		}

		QVariant
		operator()( const double & aVal ) const
		{
			return QVariant( aVal );
		}

		QVariant 
		operator()( const std::string & aVal ) const
		{
			return QVariant( aVal.data() );
		}

		template< typename TType, size_t tDim >
		QVariant 
		convertVector( const Vector< TType, tDim > & aVal ) const
		{
			std::string str = TO_STRING( aVal );
			return QVariant( str.data() );
			/*QList<QVariant> data;
			for ( size_t i = 0; i < tDim; ++i ) {
				data.push_back( QVariant( aVal[i] ) );
			}
			return QVariant( data );*/
		}

		QVariant 
		operator()( const Vector< double, 3 > & aVal ) const
		{
			return convertVector< double, 3 >( aVal );
		}

		QVariant 
		operator()( const Vector< double, 4 > & aVal ) const
		{
			return convertVector< double, 4 >( aVal );
		}

	};

	class TypeNameVisitor: public boost::static_visitor<QString>
	{
	public:
		QString
		operator()( const bool & aVal ) const
		{
			return tr( "Boolean" );
		}

		QString
		operator()( const int & aVal ) const
		{
			return tr( "Integer" );
		}

		QString
		operator()( const double & aVal ) const
		{
			return tr( "Double" );
		}

		QString 
		operator()( const std::string & aVal ) const
		{
			return tr( "String" );
		}


		QString 
		operator()( const Vector< double, 3 > & aVal ) const
		{
			return tr( "3xDouble" );
		}

		QString 
		operator()( const Vector< double, 4 > & aVal ) const
		{
			return tr( "4xDouble" );
		}

	};

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
