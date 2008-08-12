#include "qapplication.h"

#include "vtkPNGReader.h"
#include "vtkImageClip.h"

#include "myImageViewer.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"

#include "QVTKWidget.h"

// whatever

#include "vtkActor2D.h"
#include "vtkCommand.h"
#include "vtkImageData.h"
#include "vtkInteractorStyleImage.h"
#include "vtkObjectFactory.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"

#include "myCallback.h"



int main(int argc, char** argv)
{
  QApplication app(argc, argv);

  QVTKWidget widget;
  widget.resize(256,256);

 // Stages of the pipeline

 // Read a PNG
  vtkPNGReader* i0 = vtkPNGReader::New();
  i0->SetFileName("../test.png");


 // Perform a RFFT
  vtkImageClip* i3 = vtkImageClip::New();
  i3->SetInputConnection(i0->GetOutputPort());

 // View it!
  vtkImageViewer* image_view = vtkImageViewer::New();
  image_view->SetInputConnection(i3->GetOutputPort());

  widget.SetRenderWindow(image_view->GetRenderWindow());

 // Setup the interactor for it
  vtkRenderWindowInteractor *rwi = widget.GetRenderWindow()->GetInteractor();
  
  image_view->InteractorStyle = vtkInteractorStyleImage::New();
    
  myCallback *c = myCallback::New();
  c->IV = image_view;
  c->i2 = i3;
  image_view->InteractorStyle->AddObserver(vtkCommand::WindowLevelEvent, c);
  image_view->InteractorStyle->AddObserver(vtkCommand::StartWindowLevelEvent, c);
  image_view->InteractorStyle->AddObserver(vtkCommand::ResetWindowLevelEvent, c);
  c->Delete();

  image_view->Interactor = rwi;
  rwi->Register(image_view);

  image_view->Interactor->SetInteractorStyle(image_view->InteractorStyle);
  image_view->Interactor->SetRenderWindow(image_view->RenderWindow);

  image_view->SetColorLevel(138.5);
  image_view->SetColorWindow(233);

  widget.show();
  app.exec();
  
  i0->Delete();
//  i1->Delete();
//  i2->Delete();
  i3->Delete();
  image_view->Delete();

  return 0;
}
