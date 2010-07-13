#ifndef ADESKTOP_CONTAINER_FACTORY_H
#define ADESKTOP_CONTAINER_FACTORY_H

#include <QtGui>
#include <boost/shared_ptr.hpp>
#include "GUI/widgets/ADesktopContainer.h"

namespace M4D
{
namespace GUI
{


class ADesktopContainerFactory
{
public:
	typedef boost::shared_ptr< ADesktopContainerFactory > Ptr;

	ADesktopContainerFactory();

	~ADesktopContainerFactory();

	virtual ADesktopContainer *
	Create() = 0;

protected:
private:

};



} /*namespace GUI*/
} /*namespace M4D*/


#endif /*ADESKTOP_CONTAINER_FACTORY_H*/


