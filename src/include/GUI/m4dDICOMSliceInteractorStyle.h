/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file m4dDICOMSliceInteractorStyle.h 
 * @{ 
 **/

#ifndef __m4dDICOMSliceInteractorStyle_h
#define __m4dDICOMSliceInteractorStyle_h

#include "vtkRenderer.h"
#include "vtkInteractorStyle.h"
#include "m4dDICOMSliceReader.h"

class VTK_RENDERING_EXPORT m4dDICOMSliceInteractorStyle : public vtkInteractorStyle
{
  
  private:
    
    // The renderer that contains the current slice that is displayed.
    vtkRenderer *currentRenderer;
    
    // The slice reader that reads a given slice and puts it into a renderer.
    m4dDICOMSliceReader *sliceReader;
    
    // The directory of the DICOM images.
    char *currentDirectory;

    // The number of the current slice on display.
    int currentSlice;
  
  public:

    // Create a new instance.
    static m4dDICOMSliceInteractorStyle *New();

    // Interact this way if the mouse wheel is rotated forward.
    virtual void OnMouseWheelForward();

    // Interact this way if the mouse wheel is rotated backward.
    virtual void OnMouseWheelBackward();

    // Init the DICOM image path and the image that is shown in the beginning.
    // It needs to be called before any interaction could be made.
    bool initRenderer( const char* dir );

  protected:
    m4dDICOMSliceInteractorStyle();
    ~m4dDICOMSliceInteractorStyle();

  private:
    m4dDICOMSliceInteractorStyle(const m4dDICOMSliceInteractorStyle&);  // Not implemented.
    void operator=(const m4dDICOMSliceInteractorStyle&);  // Not implemented.

};

#endif

/** @} */

