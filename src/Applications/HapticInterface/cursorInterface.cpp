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
			std::cout << "making deep copy";
			this->input->DeepCopy(input);
			std::cout << "done." << std::endl;
			cursor = vtkSphereSource::New();
			cursor->SetRadius(1.0);
			cursorRadiusCube = vtkCubeSource::New();
			reloadParameters();
		}
		cursorInterface::~cursorInterface()
		{
			cursor->Delete();
			cursorRadiusCube->Delete();
		}
		void cursorInterface::reloadParameters()
		{
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
								if (minVolumeValue > (unsigned short)input->GetScalarComponentAsDouble(i,j,k,c))
									minVolumeValue = (unsigned short)input->GetScalarComponentAsDouble(i,j,k,c);
								if (maxVolumeValue < (unsigned short)input->GetScalarComponentAsDouble(i,j,k,c))
									maxVolumeValue = (unsigned short)input->GetScalarComponentAsDouble(i,j,k,c);
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

				double center[3];
				center[0] = (imageRealWidth / 2.0) + imageRealOffsetWidth;
				center[1] = (imageRealHeight / 2.0) + imageRealOffsetHeight;
				center[2] = (imageRealDepth / 2.0) + imageRealOffsetDepth;

				cursor->SetCenter(center);

				SetScale(MAX( MAX(imageRealWidth, imageRealHeight), MAX(imageRealHeight, imageRealDepth)));						
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
			double center[3];
			cursor->GetCenter(center);
			cursorRadiusCube->SetCenter(center);
			cursorRadiusCube->SetXLength(scale);
			cursorRadiusCube->SetYLength(scale);
			cursorRadiusCube->SetZLength(scale);
		}

		void cursorInterface::SetCursorPosition(const cVector3d& position)
		{
			cursor->SetCenter(position.x, position.y, position.z); // Bad implementation
		}

		vtkCubeSource* cursorInterface::GetRadiusCube()
		{
			return cursorRadiusCube;
		}

		vtkSphereSource* cursorInterface::GetCursor()
		{
			return cursor;
		}

		float cursorInterface::GetX()
		{
			double center[3];
			cursor->GetCenter(center);
			return center[0];
		}

		float cursorInterface::GetY()
		{
			double center[3];
			cursor->GetCenter(center);
			return center[1];
		}

		float cursorInterface::GetZ()
		{
			double center[3];
			cursor->GetCenter(center);
			return center[2];
		}
	}
}

#endif