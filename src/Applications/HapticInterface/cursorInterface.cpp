#include "cursorInterface.h"

M4D::Viewer::cursorInterface::cursorInterface(Imaging::InputPortTyped< Imaging::AImage > *inPort)
{
	_inPort = inPort;
	x = 0.0;
	y = 0.0;
	z = 0.0;
}

float M4D::Viewer::cursorInterface::getX()
{
	return x;
}

float M4D::Viewer::cursorInterface::getY()
{
	return y;
}

float M4D::Viewer::cursorInterface::getZ()
{
	return z;
}