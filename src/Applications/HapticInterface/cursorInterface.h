#ifndef HAPTIC_VIEWER_CURSOR_INTERFACE
#define HAPTIC_VIEWER_CURSOR_INTERFACE

#include "Imaging/Imaging.h"

namespace M4D
{
	namespace Viewer
	{
		class cursorInterface
		{
		public:
			virtual float getX();// returns value from -1.0 to 1.0 - relative position of cursor in space
			virtual float getY();
			virtual float getZ();
			cursorInterface(Imaging::InputPortTyped< Imaging::AImage >	*inPort);
		protected: 
			Imaging::InputPortTyped< Imaging::AImage >	*_inPort;
			float x;
			float y;
			float z;
		};
	}
}

#endif