#ifndef M4D_GUI_CURSORINTERFACE_H_
#define M4D_GUI_CURSORINTERFACE_H_

#define MAX(a, b) (a > b ? a : b)

#include "cursorInterface.h"

namespace M4D
{
	namespace Viewer
	{
		cursorInterface::cursorInterface(Imaging::InputPortTyped< Imaging::AImage >* inPort)
		{
			this->inPort = inPort;
			reloadParameters();
		}
		void cursorInterface::reloadParameters()
		{
			if (inPort->GetDatasetTyped().GetDimension() == 3)
			{
				imageDataWidth = inPort->GetDatasetTyped().GetDimensionExtents(0).maximum - inPort->GetDatasetTyped().GetDimensionExtents(0).minimum;
				imageDataHeight = inPort->GetDatasetTyped().GetDimensionExtents(1).maximum - inPort->GetDatasetTyped().GetDimensionExtents(1).minimum;
				imageDataDepth = inPort->GetDatasetTyped().GetDimensionExtents(2).maximum - inPort->GetDatasetTyped().GetDimensionExtents(2).minimum;

				imageRealWidth = ((float)imageDataWidth) * inPort->GetDatasetTyped().GetDimensionExtents(0).elementExtent;
				imageRealHeight = ((float)imageDataHeight) * inPort->GetDatasetTyped().GetDimensionExtents(1).elementExtent;
				imageRealDepth = ((float)imageDataDepth) * inPort->GetDatasetTyped().GetDimensionExtents(2).elementExtent;

				cursorPosition.x = imageRealWidth / 2.0;
				cursorPosition.y = imageRealHeight / 2.0;
				cursorPosition.z = imageRealDepth / 2.0;

				cubeCenter = cursorPosition;
				
				scale = MAX( MAX(imageRealWidth, imageRealHeight), MAX(imageRealHeight, imageRealDepth)); 							
			}
			else
			{
				scale = -1.0;
			}
		}

		double cursorInterface::GetScale()
		{
			return scale;
		}

		void cursorInterface::SetScale(double scale)
		{
			this->scale = scale;
			cubeCenter = cursorPosition;
		}

		const cVector3d& cursorInterface::GetCubeCenter()
		{
			return cubeCenter;
		}

		void cursorInterface::SetCursorPosition(cVector3d& cursorPosition)
		{
			this->cursorPosition = cursorPosition;
		}

		const cVector3d& cursorInterface::GetCursorPosition()
		{
			return cursorPosition;
		}

		float cursorInterface::GetX()
		{
			return cursorPosition.x;
		}

		float cursorInterface::GetY()
		{
			return cursorPosition.y;
		}

		float cursorInterface::GetZ()
		{
			return cursorPosition.z;
		}
	}
}

#endif