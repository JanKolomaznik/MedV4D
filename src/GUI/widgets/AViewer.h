#ifndef AVIEWER_H
#define AVIEWER_H

#include "common/Common.h"
#include "GUI/utils/AHUDInfo.h"

namespace M4D
{
namespace GUI
{
namespace Viewer
{


enum RenderingQuality
{
	rqLowest,
	rqLow,
	rqNormal,
	rqHigh,
	rqHighest
};


class AViewer
{

public:
	AViewer()
	{
		_id = ++AViewer::_lastID;
	}

	virtual ~AViewer() {}


	/**
	* Return the list of input ports connected to this viewer.
	*  @return list of connected input ports
	*/
	const Imaging::InputPortList &
	InputPort()const
	{ return _inputPorts; }

	/**
	* List of the input ports connected to the given viewer.
	*/
	Imaging::InputPortList				_inputPorts;

protected:

	static int32	_lastID;

	int32		_id;

	HUDInfoList	_hudInfoList;
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
