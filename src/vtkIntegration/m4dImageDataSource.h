#ifndef _M4D_IMAGE_DATA_SOURCE_H
#define _M4D_IMAGE_DATA_SOURCE_H

/**
 *  @ingroup vtkintegration
 *  @file DataConversion.h
 *  @{
 */

#include "common/ExceptionBase.h"
#include "Imaging/AbstractImageData.h"
#include "Imaging/Image.h"

#include "vtkImageAlgorithm.h"

namespace M4D
{
namespace vtkIntegration
{

class m4dImageDataSource : public vtkImageAlgorithm
{
public:
	static m4dImageDataSource *New();

	vtkTypeRevisionMacro(m4dImageDataSource,vtkImageAlgorithm);

	void
	SetImageData( Imaging::AbstractImage::Ptr imageData );

	void
	TemporarySetImageData( const Imaging::AbstractImage & imageData );

	void
	TemporaryUnsetImageData();
protected:
	m4dImageDataSource();
	~m4dImageDataSource();

	int 
	RequestInformation (
		vtkInformation 		*,
		vtkInformationVector	**,
		vtkInformationVector	*
		);

	int 
	RequestData(
		vtkInformation *, 
		vtkInformationVector **, 
		vtkInformationVector *
		);

	Imaging::AbstractImage::Ptr	_imageData;
	const Imaging::AbstractImage		*_tmpImageData;

	int				_wholeExtent[6];
	double				_spacing[3];
private:
	m4dImageDataSource(const m4dImageDataSource&);  // Not implemented.
	void operator=(const m4dImageDataSource&);  // Not implemented.


};

} /*namespace vtkIntegration*/
} /*namespace M4D*/

/** @} */

#endif /*_M4D_IMAGE_DATA_SOURCE_H*/
