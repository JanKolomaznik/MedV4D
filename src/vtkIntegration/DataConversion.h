#ifndef _DATA_CONVERSION_H
#define _DATA_CONVERSION_H

#include "ImageDataTemplate.h"

#include <vtkImageData>

namespace M4D
{

namespace vtkIntegration
{

template< typename ElementType >
void
SetVTKImageDataScalarType( vtkImageData & );


/**
 * Creates new instance of vtkImageData containing copy of image passed as argument.
 * @param image Instance of image, which should be converted to VTK representation.
 * @exception None
 * @return New instance of vtkImageData.
 **/
template< typename ElementType >
vtkImageData*
CreateVTKImageDataFromImageData( const Images::ImageDataTemplate< ElementType >& image )
{
	size_t width, height, depth;
	vtkIdType IncX, IncY, IncZ;
	vtkImageData* imageData = vtkImageData::New();

	imageData->SetSpacing(voxelsize.x, voxelsize.y, voxelsize.z);
	imageData->SetDimensions(width, height, depth);
	//TODO Exception handling
	SetVTKImageDataScalarType( *imageData );

	imageData->GetIncrements(IncX, IncY, IncZ);

	ElementType* iPtr = (ElementType*)imageData->GetScalarPointer();

	for(int idxZ = 0; idxZ < d; idxZ++)
	{
		for(int idxY = 0; idxY < h; idxY++)
		{
			for(int idxX = 0; idxX < w; idxX++)
			{
				TVector4 vox = voxel2data.Transform(idxX, idxY, idxZ).Normalize();
				int x = ROUND(vox.x);
				int y = ROUND(vox.y);
				int z = ROUND(vox.z);
				*iPtr = dataset->voxels.data[z][y][x];
				iPtr++;
			}
			iPtr += IncY;
		}
		iPtr += IncZ;
	}
	return imageData;
}


}/*namespace vtkIntegration*/
}/*namespace M4D*/

#endif /*_DATA_CONVERSION_H*/
