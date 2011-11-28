#ifndef PARAMETER_SET_H
#define PARAMETER_SET_H

#include <map>
#include <string>
#include <algorithm>
#include <boost/any.hpp>
#include "MedV4D/Common/ExceptionBase.h"

class ParameterSet
{
public:
	typedef std::map< std::string, boost::any >::const_iterator ConstIterator;
	typedef std::map< std::string, boost::any >::iterator Iterator;


	ConstIterator
	Begin()const
	{ return _parameters.begin(); }
	
	Iterator
	Begin()
	{ return _parameters.begin(); }

	ConstIterator
	End()const
	{ return _parameters.end(); }
	
	Iterator
	End()
	{ return _parameters.end(); }

	size_t
	Size()const
	{ return _parameters.size(); }

	void
	Add( const std::string &name, const boost::any &value )
	{
		ConstIterator it = _parameters.find( name );
		if ( it == _parameters.end() ) {
			_parameters[ name ] = value;
		}
		_THROW_ M4D::ErrorHandling::EAlreadyPresent( TO_STRING( "Parameter '" << name << "' already present!" ) );
	}

	const boost::any &
	operator[]( const std::string &name ) const
	{
		ConstIterator it = _parameters.find( name );
		if ( it != _parameters.end() ) {
			return it->second;
		}
		_THROW_ M4D::ErrorHandling::ENotFound( TO_STRING( "Parameter '" << name << "' not found!" ) );
	}

	boost::any &
	operator[]( const std::string &name )
	{
		Iterator it = _parameters.find( name );
		if ( it != _parameters.end() ) {
			return it->second;
		}
		_THROW_ M4D::ErrorHandling::ENotFound( TO_STRING( "Parameter '" << name << "' not found!" ) );
	}
protected:
	std::map< std::string, boost::any >	_parameters;
private:

};



#endif /*PARAMETER_SET_H*/
