#ifndef AVIEWER_H
#define AVIEWER_H

#include "common/Common.h"

namespace M4D
{
namespace Viewer
{



class AViewer
{

public:
	AViewer()
	{
		_id = ++AViewer::_lastID;
	}

	virtual ~AViewer() {}

protected:

	static int32	_lastID;

	int32		_id;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AViewer );
};


} /*namespace Viewer*/
} /*namespace M4D*/


#endif /*AVIEWER_H*/
