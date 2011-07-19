#ifndef MODULE_H
#define MODULE_H

#include "common/Common.h"
#include <map>
#include <boost/shared_ptr.hpp>


class AModule
{
public:
	typedef boost::shared_ptr< AModule > Ptr;

	virtual void
	load() = 0;

	virtual void
	unload() = 0;

	virtual bool
	isUnloadable() = 0;

	virtual bool
	isLoaded() = 0;

	virtual std::string
	getName() = 0;
};

typedef std::map< std::string, AModule::Ptr > ModuleMap;

#endif /*MODULE_H*/
