#ifndef TESTDATAGENERATOR_H_
#define TESTDATAGENERATOR_H_

#include "Imaging/ImageFactory.h"

M4D::Imaging::Image<uint16, 2>::Ptr
CreateTestImage(uint32 width, uint32 height)
{	
	M4D::Imaging::Image<uint16, 2>::Ptr ds = 
		M4D::Imaging::ImageFactory::CreateEmptyImage2DTyped< uint16 >(width, height);

	int32 xStride;
	int32 yStride;
	uint16 *pointer = ds->GetPointer( width, height, xStride, yStride );

	uint32 fieldSize = width / 8;
	uint32 xNum, yNum;
	yNum = 1;
	for(uint32 j=1; j<=height; j++)
	{
		xNum = yNum;
		for(uint32 i=1; i<=width; i++)
		{
			*pointer = xNum;
			pointer++;
			if((i % fieldSize) == 0)
				xNum ++;
		}
		if((j % fieldSize) == 0)
			yNum += 8;
	}
	return ds;
}

#endif /*TESTDATAGENERATOR_H_*/
