#pragma once

#include <QtWidgets>
#include <boost/shared_ptr.hpp>
#include "MedV4D/GUI/widgets/ADesktop.h"
#include "MedV4D/GUI/widgets/ADesktopContainer.h"
#include "MedV4D/GUI/widgets/utils/ADesktopContainerFactory.h"

namespace M4D
{
namespace GUI
{


class Desktop: public QWidget
{
	Q_OBJECT;
public:
	Desktop( QWidget *parent = NULL );

	~Desktop();

	virtual void
	SetContainerFactory( ADesktopContainerFactory::Ptr factory );

protected:
	std::vector< ADesktopContainer * >	_desktopContainers;

	ADesktopContainerFactory::Ptr		_containerFactory;
private:

};



} /*namespace GUI*/
} /*namespace M4D*/



