#ifndef MODULE_H
#define MODULE_H

#include "common/Common.h"
#include <map>
#include <boost/shared_ptr.hpp>


class AModule
{
public:
	typedef boost::shared_ptr< AModule > Ptr;

	AModule(): mLoaded( false )
	{}

	virtual void
	load() = 0;

	virtual void
	unload() = 0;

	virtual bool
	isUnloadable() = 0;

	virtual bool
	isLoaded()
	{
		return mLoaded;
	}

	virtual std::string
	getName() = 0;
protected:
	bool mLoaded;
};

template< typename TModule >
AModule::Ptr
createModule()
{
	return AModule::Ptr( new TModule );
}


typedef std::map< std::string, AModule::Ptr > ModuleMap;

#endif /*MODULE_H*/
