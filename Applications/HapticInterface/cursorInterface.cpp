#ifndef M4D_GUI_CURSORINTERFACE_H_
#define M4D_GUI_CURSORINTERFACE_H_

#define MAX(a, b) (a > b ? a : b)

#include "cursorInterface.h"

namespace M4D
{
	namespace Viewer
	{
		cursorInterface::cursorInterface(vtkImageData* input)
		{
			this->input = vtkImageData::New();
			this->input->DeepCopy(input);
			cursorCenter[0] = 0.0;
			cursorCenter[1] = 0.0;
			cursorCenter[2] = 0.0;
			cursorRadiusCubeCenter[0] = 0.0;
			cursorRadiusCubeCenter[1] = 0.0;
			cursorRadiusCubeCenter[2] = 0.0;
			reloadParameters();
		}
		cursorInterface::cursorInterface()
		{
			this->input = vtkImageData::New();
			cursorCenter[0] = 0.0;
			cursorCenter[1] = 0.0;
			cursorCenter[2] = 0.0;
			cursorRadiusCubeCenter[0] = 0.0;
			cursorRadiusCubeCenter[1] = 0.0;
			cursorRadiusCubeCenter[2] = 0.0;
			reloadParameters();
		}
		void cursorInterface::SetData( vtkImageData* a_input)
		{
			{
				boost::mutex::scoped_lock l(cursorMutex);
				if (input)
				{
					input->Delete();
				}
				this->input = vtkImageData::New();
				this->input->DeepCopy(a_input);
				cursorCenter[0] = 0.0;
				cursorCenter[1] = 0.0;
				cursorCenter[2] = 0.0;
				cursorRadiusCubeCenter[0] = 0.0;
				cursorRadiusCubeCenter[1] = 0.0;
				cursorRadiusCubeCenter[2] = 0.0;
			}
			reloadParameters();
		}
		void cursorInterface::reloadParameters()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			if (input->GetDataDimension() == 3)
			{
				int dimensions[3];
				input->GetDimensions(dimensions);

				double spacing[3];
				input->GetSpacing(spacing);

				int extents[6];
				input->GetExtent(extents);
				
				minVolumeValue = (unsigned short)input->GetScalarTypeMax();
				maxVolumeValue = (unsigned short)input->GetScalarTypeMin();
				for (int i = extents[0]; i < extents[1]; ++i)
				{
					for (int j = extents[2]; j < extents[3]; ++j)
					{
						for (int k = extents[4]; k < extents[5]; ++k)
						{
							for (int c = 0; c < input->GetNumberOfScalarComponents(); ++c)
							{
								unsigned short result = (unsigned short)input->GetScalarComponentAsDouble(i,j,k,c);
								if (minVolumeValue > result)
									minVolumeValue = result;
								if (maxVolumeValue < result)
									maxVolumeValue = result;
							}
						}
					}
				}

				imageSpacingWidth = spacing[0];
				imageSpacingHeight = spacing[1];
				imageSpacingDepth = spacing[2];

				imageOffsetWidth = extents[0];
				imageOffsetHeight = extents[2];
				imageOffsetDepth = extents[4];
				
				imageRealOffsetWidth = extents[0] * spacing[0];
				imageRealOffsetHeight = extents[2] * spacing[1];
				imageRealOffsetDepth = extents[4] * spacing[2];

				imageDataWidth = dimensions[0];
				imageDataHeight = dimensions[1];
				imageDataDepth = dimensions[2];

				imageRealWidth = imageDataWidth * spacing[0];
				imageRealHeight = imageDataHeight * spacing[1];
				imageRealDepth = imageDataDepth * spacing[2];

				cursorCenter[0] = (imageRealWidth / 2.0) + imageRealOffsetWidth;
				cursorCenter[1] = (imageRealHeight / 2.0) + imageRealOffsetHeight;
				cursorCenter[2] = (imageRealDepth / 2.0) + imageRealOffsetDepth;

				this->scale = MAX( MAX(imageRealWidth, imageRealHeight), MAX(imageRealHeight, imageRealDepth));
				originalScale = scale;

				cursorRadiusCubeCenter[0] = cursorCenter[0];
				cursorRadiusCubeCenter[1] = cursorCenter[1];
				cursorRadiusCubeCenter[2] = cursorCenter[2];

				cursorRadiusCubeCenterOriginal[0] = cursorCenter[0];
				cursorRadiusCubeCenterOriginal[1] = cursorCenter[1];
				cursorRadiusCubeCenterOriginal[2] = cursorCenter[2];
			}
			else
			{
				scale = -1.0;
			}
		}

		double cursorInterface::GetScale()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			return scale;
		}

		void cursorInterface::SetScale(double scale)
		{
			boost::mutex::scoped_lock lck(cursorMutex);

			double positionDiff[3];
			positionDiff[0] = cursorCenter[0] - cursorRadiusCubeCenter[0]; 
			positionDiff[1] = cursorCenter[1] - cursorRadiusCubeCenter[1];
			positionDiff[2] = cursorCenter[2] - cursorRadiusCubeCenter[2];
			
			double scaleRatio = scale / this->scale;
			this->scale = scale;
			cursorRadiusCubeCenter[0] = cursorCenter[0];// + positionDiff[0] * scaleRatio;
			cursorRadiusCubeCenter[1] = cursorCenter[1];// + positionDiff[1] * scaleRatio;
			cursorRadiusCubeCenter[2] = cursorCenter[2];// + positionDiff[2] * scaleRatio;
		}

		void cursorInterface::SetCursorPosition(const cVector3d& position)
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			cursorCenter[0] = position.x;
			cursorCenter[1] = position.y;
			cursorCenter[2] = position.z;
		}

		void cursorInterface::GetRadiusCubeCenter(double center[3])
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			center[0] = cursorRadiusCubeCenter[0];
			center[1] = cursorRadiusCubeCenter[1];
			center[2] = cursorRadiusCubeCenter[2];
		}

		void cursorInterface::GetCursorCenter( double center[3] )
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			center[0] = cursorCenter[0];
			center[1] = cursorCenter[1];
			center[2] = cursorCenter[2];
		}

		double cursorInterface::GetX()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			return cursorCenter[0];
		}

		double cursorInterface::GetY()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			return cursorCenter[1];
		}

		double cursorInterface::GetZ()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			return cursorCenter[2];
		}

		int cursorInterface::GetZSlice()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			return (int)(cursorCenter[2] / imageSpacingDepth);
		}

		void cursorInterface::GetCursorCenterAsIndexes( int center[3] )
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			center[0] = (int)(cursorCenter[0] / imageSpacingWidth); 
			center[1] = (int)(cursorCenter[1] / imageSpacingHeight); 
			center[2] = (int)(cursorCenter
				[2] / imageSpacingDepth);
		}

		void cursorInterface::GetRadiusCubeCenterAsIndexes( int center[3] )
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			center[0] = (int)(cursorRadiusCubeCenter[0] / imageSpacingWidth); 
			center[1] = (int)(cursorRadiusCubeCenter[1] / imageSpacingHeight); 
			center[2] = (int)(cursorRadiusCubeCenter[2] / imageSpacingDepth); 
		}

		int cursorInterface::GetScaleAsIndexDifference()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			double maxRealLength = MAX( MAX(imageRealWidth, imageRealHeight), MAX(imageRealHeight, imageRealDepth));
			int maxDataLength = MAX( MAX(imageDataWidth, imageDataHeight), MAX(imageDataHeight, imageDataDepth));
			return (int)(( scale / maxRealLength) * maxDataLength);
		}

		int cursorInterface::GetDataMinValue()
		{
			return minVolumeValue;
		}

		int cursorInterface::GetDataMaxValue()
		{
			return maxVolumeValue;
		}

		void cursorInterface::SetToOriginal()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			scale = originalScale;
			cursorRadiusCubeCenter[0] = cursorRadiusCubeCenterOriginal[0];
			cursorRadiusCubeCenter[1] = cursorRadiusCubeCenterOriginal[1];
			cursorRadiusCubeCenter[2] = cursorRadiusCubeCenterOriginal[2];
		}
	}
}

#endif