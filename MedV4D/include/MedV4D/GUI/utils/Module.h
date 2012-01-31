#ifndef MODULE_H
#define MODULE_H

#include "MedV4D/Common/Common.h"
#include <map>
#include <boost/shared_ptr.hpp>


class AModule
{
public:
	typedef boost::shared_ptr< AModule > Ptr;

	AModule( std::string aName ): mLoaded( false ), mName( aName )
	{
		LOG( "Constructing module" );
	}

	~AModule()
	{
		LOG( "Destroying \"" << getName() << "\"" );
	}
	
	void
	load()
	{
		LOG_CONT( "Loading \"" << getName() << "\"" );
		loadModule();
		LOG( "\t\tDONE" );
	}
	

	void
	unload()
	{
		LOG_CONT( "Unloading \"" << getName() << "\"" );
		unloadModule();
		LOG( "\t\tDONE" );
	}

	virtual bool
	isUnloadable() = 0;

	virtual bool
	isLoaded()
	{
		return mLoaded;
	}

	std::string
	getName()
	{
		return mName;
	}
protected:
	virtual void
	loadModule() = 0;
	
	virtual void
	unloadModule() = 0;
	
	bool mLoaded;
	std::string mName;
};

template< typename TModule >
AModule::Ptr
createModule()
{
	return AModule::Ptr( new TModule );
}


typedef std::map< std::string, AModule::Ptr > ModuleMap;

#endif /*MODULE_H*/
