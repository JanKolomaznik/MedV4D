#ifndef ADESKTOP_CONTAINER_FACTORY_H
#define ADESKTOP_CONTAINER_FACTORY_H

#include <QtGui>
#include <boost/shared_ptr.hpp>
#include "MedV4D/GUI/widgets/ADesktopContainer.h"
#include "MedV4D/Common/ParameterSet.h"

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

	virtual ADesktopContainer *
	Create( const ParameterSet &info ) = 0;

protected:
private:

};



} /*namespace GUI*/
} /*namespace M4D*/


#endif /*ADESKTOP_CONTAINER_FACTORY_H*/


