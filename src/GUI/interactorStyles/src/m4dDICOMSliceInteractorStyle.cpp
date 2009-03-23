#include "m4dDICOMSliceInteractorStyle.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderWindow.h"
#include "string.h"

m4dDICOMSliceInteractorStyle* m4dDICOMSliceInteractorStyle::New()
{
  return new m4dDICOMSliceInteractorStyle();
}

void m4dDICOMSliceInteractorStyle::OnMouseWheelForward()
{
  currentSlice++;
  vtkRenderer *ren = sliceReader->getSlice( currentDirectory, currentSlice );
  if ( ren )
  {
    if ( currentRenderer ) GetInteractor()->GetRenderWindow()->RemoveRenderer(currentRenderer);
    currentRenderer = ren;
    GetInteractor()->GetRenderWindow()->AddRenderer(currentRenderer);
    GetInteractor()->GetRenderWindow()->Render();
  }
}

void m4dDICOMSliceInteractorStyle::OnMouseWheelBackward()
{
  currentSlice--;
  vtkRenderer *ren = sliceReader->getSlice( currentDirectory, currentSlice );
  if ( ren )
  {
    if ( currentRenderer ) GetInteractor()->GetRenderWindow()->RemoveRenderer(currentRenderer);
    currentRenderer = ren;
    GetInteractor()->GetRenderWindow()->AddRenderer(currentRenderer);
    GetInteractor()->GetRenderWindow()->Render();
  }
}

bool m4dDICOMSliceInteractorStyle::initRenderer( const char* dir )
{
  vtkRenderer *ren = sliceReader->getSlice( dir, currentSlice );
  if ( ren )
  {
    currentDirectory = new char[ strlen(dir) + 1 ];
    strncpy(currentDirectory,dir,strlen(dir) + 1);
    if ( currentRenderer ) GetInteractor()->GetRenderWindow()->RemoveRenderer(currentRenderer);
    currentRenderer = ren;
    GetInteractor()->GetRenderWindow()->AddRenderer(currentRenderer);
    GetInteractor()->GetRenderWindow()->Render();
    return true;
  }
  return false;
}

m4dDICOMSliceInteractorStyle::m4dDICOMSliceInteractorStyle()
{
  currentSlice = 0;
  sliceReader = m4dDICOMSliceReader::New();
  currentRenderer = 0;
}

m4dDICOMSliceInteractorStyle::~m4dDICOMSliceInteractorStyle()
{
}
