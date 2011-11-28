#ifndef AVIEWER_H
#define AVIEWER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/IDGenerator.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{



class AViewer
{

public:
	AViewer()
	{
		_id = AViewer::_IDGenerator.NewID();
	}

	virtual 
	~AViewer() 
	{}

protected:

	static M4D::Common::IDGenerator	_IDGenerator;

	int32				_id;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AViewer );
};


} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


#endif /*AVIEWER_H*/
