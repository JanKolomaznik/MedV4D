#pragma once

#include "MedV4D/GUI/utils/Module.h"

#include "DatasetManager.hpp"

class ViewerModule : public AModule
{
public:
	ViewerModule(std::string aName)
		: AModule(std::move(aName))
	{}

	virtual void
	setDatasetManager(DatasetManager &aDatasetManager) = 0;

};
