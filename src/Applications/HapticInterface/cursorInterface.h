#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_CURSOR_INTERFACE
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_CURSOR_INTERFACE
#define _MSVC

#include "Imaging/Imaging.h"
#include "chai3d.h"
#include "vtkSphereSource.h"
#include "vtkCubeSource.h"
#include "vtkImageData.h"

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
			virtual vtkSphereSource* GetCursor(); // returns cursor position as vector
			virtual vtkCubeSource* GetRadiusCube(); // returns cube center position as vector
			virtual double GetScale(); // returns size of cube where is action radius of cursor
			virtual void reloadParameters(); // reload image parameters from inPort
			cursorInterface(vtkImageData* input);
			~cursorInterface();
		protected: 
			virtual void SetCursorPosition(const cVector3d& position);
			virtual void SetScale(double scale); // Sets scale
			vtkImageData* input; // link to dataset
			vtkSphereSource* cursor;
			vtkCubeSource* cursorRadiusCube;
			unsigned short minVolumeValue, maxVolumeValue;
			double scale; // size of cube where is action radius of cursor
			double imageRealHeight, imageRealWidth, imageRealDepth; // parameters of volume dataset - size in mm
			double imageRealOffsetHeight, imageRealOffsetWidth, imageRealOffsetDepth; // offset which indicates how far VTK starts drawing of object from 0,0,0
			double imageSpacingWidth, imageSpacingHeight, imageSpacingDepth; // spacing of eachdimension
			int imageOffsetHeight, imageOffsetWidth, imageOffsetDepth;
			int imageDataHeight, imageDataWidth, imageDataDepth; // parameters of volume dataset - size in voxels
		};
	}
}

#endif