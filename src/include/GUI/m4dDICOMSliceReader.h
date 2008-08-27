/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file m4dDICOMSliceReader.h 
 * @{ 
 **/

#ifndef __m4dDICOMSliceReader_h
#define __m4dDICOMSliceReader_h

#include "vtkDICOMImageReader.h"
#include "vtkRenderer.h"

class m4dDICOMSliceReader : public vtkDICOMImageReader
{

  public:

    // Create a new instance.
    static m4dDICOMSliceReader* New();

    // Get the slice with number "slice" from the images located in directory "dir".
    // If slice number is too small, it will be modified to 0; if it is too large,
    // it will be modified to the number of the last image slice.
    vtkRenderer* getSlice( const char* dir, int& slice );

  protected:
    m4dDICOMSliceReader();
    ~m4dDICOMSliceReader();

  private:
    m4dDICOMSliceReader(const m4dDICOMSliceReader&); // Not implemented.
    void operator=(const m4dDICOMSliceReader&); // Not implemented.

};

#endif

/** @} */

