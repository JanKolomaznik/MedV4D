#include "m4dDICOMSliceReader.h"
#include "vtkImageCast.h"
#include "vtkImageActor.h"

m4dDICOMSliceReader* m4dDICOMSliceReader::New()
{
  return new m4dDICOMSliceReader();
}

vtkRenderer* m4dDICOMSliceReader::getSlice( const char* dir, int& slice )
{
  if ( !dir ) return NULL;
  
  // Set the directory of the DICOM images.
  this->SetDirectoryName( dir );
  this->ExecuteInformation();
  
  // Check whether slice number is out of bounds.
  if ( slice >= this->GetNumberOfDICOMFileNames() ) slice = this->GetNumberOfDICOMFileNames() - 1;
  if ( slice < 0 ) slice = 0;

  // Get the filename of the given slice.
  const char* file = GetDICOMFileName(slice);
  if ( !file ) return NULL;
  
  this->SetFileName( file );
  
  // Image cast to use unsigned char output.
  vtkImageCast *icast = vtkImageCast::New();
  icast->SetOutputScalarTypeToUnsignedChar();
  icast->SetInputConnection( this->GetOutputPort() );

  // Visualize the image.
  vtkImageActor* imageActor = vtkImageActor::New();
  imageActor->SetInput( icast->GetOutput() );

  // Put it into a renderer that can be returned.
  vtkRenderer *sliceRenderer = vtkRenderer::New();
  sliceRenderer->AddViewProp( imageActor );
  return sliceRenderer;
}

m4dDICOMSliceReader::m4dDICOMSliceReader()
{
}

m4dDICOMSliceReader::~m4dDICOMSliceReader()
{
}
