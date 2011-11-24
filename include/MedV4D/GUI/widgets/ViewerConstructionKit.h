#ifndef _VIEWER_CONSTRUCTION_KIT_H
#define _VIEWER_CONSTRUCTION_KIT_H

#include "MedV4D/GUI/widgets/AGUIViewer.h"
#include "MedV4D/GUI/utils/PortInterfaceHelper.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{

template< typename SupportWidget, typename PortInterface/*, template<typename> DataProcessor*/ >
class ViewerConstructionKit: public /*DataProcessor<SupportWidget>*/SupportWidget, public PortInterface, public AGUIViewer
{
public:
	QWidget* 
	CastToQWidget()
		{ return static_cast< SupportWidget * >( this ); }


protected:
	ViewerConstructionKit( QWidget * parent = NULL ): SupportWidget( parent )
	{}

private:

};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#endif /*_VIEWER_CONSTRUCTION_KIT_H*/
