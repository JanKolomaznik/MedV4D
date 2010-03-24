#ifndef HAPTIC_VIEWER_CURSOR_INTERFACE
#define HAPTIC_VIEWER_CURSOR_INTERFACE
#define _MSVC

#include "Imaging/Imaging.h"
#include "chai3d.h"

namespace M4D
{
	namespace Viewer
	{
		class cursorInterface
		{
		public:
			virtual float GetX(); // returns X part of coordinates of cursor
			virtual float GetY(); // returns Y part of coordinates of cursor
			virtual float GetZ(); // returns Z part of coordinates of cursor
			virtual const cVector3d& GetCursorPosition(); // returns cursor position as vector
			virtual const cVector3d& GetCubeCenter(); // returns cube center position as vector
			virtual double GetScale(); // returns size of cube where is action radius of cursor
			virtual void reloadParameters(); // reload image parameters from inPort
			cursorInterface(Imaging::InputPortTyped< Imaging::AImage >*	inPort);
		protected: 
			virtual void SetCursorPosition(cVector3d& cursorPosition);
			virtual void SetScale(double scale); // Sets scale
			Imaging::InputPortTyped< Imaging::AImage >*	inPort; // link to dataset
			cVector3d cursorPosition; // position of cursor
			cVector3d cubeCenter; // center of cube where is action radius of cursor
			double scale; // size of cube where is action radius of cursor
			float imageRealHeight, imageRealWidth, imageRealDepth; // parameters of volume dataset - size in mm
			int imageDataHeight, imageDataWidth, imageDataDepth; // parameters of volume dataset - size in voxels
		};
	}
}

#endif