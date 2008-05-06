#include "m4dGUIVtkRenderWindowWidget.h"

#include <QtGui>

// VTK includes - for sphereToRenderWindow and dicomToRenderWindow
// for testing purposes only
#include "vtkSphereSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkProperty.h"
#include "vtkRendererCollection.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkDICOMImageReader.h"
#include "vtkVolume.h"
#include "vtkVolumeRayCastMapper.h"
#include "vtkVolumeRayCastCompositeFunction.h"
#include "vtkImageCast.h"
#include "vtkPiecewiseFunction.h"
#include "vtkColorTransferFunction.h"
#include "vtkVolumeProperty.h"

#include "m4dImageDataSource.h"

using namespace M4D::vtkIntegration;


m4dGUIVtkRenderWindowWidget::m4dGUIVtkRenderWindowWidget ( QVTKWidget *parent )
  : QVTKWidget( parent )
{}


void m4dGUIVtkRenderWindowWidget::addRenderer ( vtkRenderer *renderer )
{ 
  vtkRenderWindow *rWin;
  rWin = GetRenderWindow();
  rWin->AddRenderer( renderer );

  vtkRenderWindowInteractor *iren;
  iren = GetInteractor();
  iren->SetRenderWindow( rWin );
}


void m4dGUIVtkRenderWindowWidget::removeFirstRenderer ()
{ 
  vtkRenderWindow *rWin;
  rWin = GetRenderWindow();
  rWin->RemoveRenderer( rWin->GetRenderers()->GetFirstRenderer() );
}


vtkRenderer *m4dGUIVtkRenderWindowWidget::imageDataToRenderWindow ()
{
  m4dImageDataSource *imageData = m4dImageDataSource::New();

  vtkImageCast *icast = vtkImageCast::New(); 
  icast->SetOutputScalarTypeToUnsignedShort();
  icast->SetInputConnection( imageData->GetOutputPort() );

  vtkPiecewiseFunction *opacityTransferFunction = vtkPiecewiseFunction::New();
  opacityTransferFunction->AddPoint( 0,   0.0 ); 	
  opacityTransferFunction->AddPoint( 169, 0.0 );
  opacityTransferFunction->AddPoint( 170, 0.2 );	
  opacityTransferFunction->AddPoint( 400, 0.2 );
  opacityTransferFunction->AddPoint( 401, 0.0 );

  vtkColorTransferFunction *colorTransferFunction = vtkColorTransferFunction::New();
  colorTransferFunction->AddRGBPoint(    0.0, 0.0, 0.0, 0.0 ); 
  colorTransferFunction->AddRGBPoint(  170.0, 1.0, 0.0, 0.0 );
  colorTransferFunction->AddRGBPoint(  400.0, 0.8, 0.8, 0.8 );
  colorTransferFunction->AddRGBPoint( 2000.0, 1.0, 1.0, 1.0 );
  	
  vtkVolumeProperty *volumeProperty = vtkVolumeProperty::New();
  volumeProperty->SetColor( colorTransferFunction );
  volumeProperty->SetScalarOpacity( opacityTransferFunction );
  volumeProperty->ShadeOn(); 
  volumeProperty->SetInterpolationTypeToLinear();

  vtkVolumeRayCastMapper *volumeMapper = vtkVolumeRayCastMapper::New();
  volumeMapper->SetVolumeRayCastFunction( vtkVolumeRayCastCompositeFunction::New());
  volumeMapper->SetInputConnection( icast->GetOutputPort() );

  vtkVolume *volume = vtkVolume::New();
  volume->SetMapper( volumeMapper ); 
  volume->SetProperty( volumeProperty );

  vtkRenderer *renImageData = vtkRenderer::New(); 
  renImageData->AddViewProp( volume );

  return renImageData;
}


// just for testing, won't be in real application
vtkRenderer *m4dGUIVtkRenderWindowWidget::sphereToRenderWindow ()
{
  // create sphere geometry
  vtkSphereSource *sphere = vtkSphereSource::New();
  sphere->SetRadius( 1.0 );
  sphere->SetThetaResolution( 8 );
  sphere->SetPhiResolution( 8 );

  // map to graphics library
  vtkPolyDataMapper *map = vtkPolyDataMapper::New();
  map->SetInput( sphere->GetOutput() );

  // actor coordinates geometry, properties, transformation
  vtkActor *aSphere = vtkActor::New();
  aSphere->SetMapper( map );
  aSphere->GetProperty()->SetColor( 1, 0, 0 );	// sphere color blue

  // a renderer and render window
  vtkRenderer *renSphere = vtkRenderer::New();
  // add the actor to the scene
  renSphere->AddActor( aSphere );
  renSphere->SetBackground( 1, 1, 1 );				// background color white

  return renSphere;
}


// just for testing, won't be in real application
vtkRenderer *m4dGUIVtkRenderWindowWidget::dicomToRenderWindow ( const char *dirName )
{
  vtkDICOMImageReader *reader = vtkDICOMImageReader::New();
  reader->SetDirectoryName( dirName ); 
  reader->SetDataSpacing( 1.0, 1.0, 3.0 );

  vtkImageCast *icast = vtkImageCast::New(); 
  icast->SetOutputScalarTypeToUnsignedShort();
  icast->SetInputConnection( reader->GetOutputPort() );

  vtkPiecewiseFunction *opacityTransferFunction = vtkPiecewiseFunction::New();
  opacityTransferFunction->AddPoint( 0,   0.0 ); 	
  opacityTransferFunction->AddPoint( 169, 0.0 );
  opacityTransferFunction->AddPoint( 170, 0.2 );	
  opacityTransferFunction->AddPoint( 400, 0.2 );
  opacityTransferFunction->AddPoint( 401, 0.0 );

  vtkColorTransferFunction *colorTransferFunction = vtkColorTransferFunction::New();
  colorTransferFunction->AddRGBPoint(    0.0, 0.0, 0.0, 0.0 ); 
  colorTransferFunction->AddRGBPoint(  170.0, 1.0, 0.0, 0.0 );
  colorTransferFunction->AddRGBPoint(  400.0, 0.8, 0.8, 0.8 );
  colorTransferFunction->AddRGBPoint( 2000.0, 1.0, 1.0, 1.0 );
  	
  vtkVolumeProperty *volumeProperty = vtkVolumeProperty::New();
  volumeProperty->SetColor( colorTransferFunction );
  volumeProperty->SetScalarOpacity( opacityTransferFunction );
  volumeProperty->ShadeOn(); 
  volumeProperty->SetInterpolationTypeToLinear();

  vtkVolumeRayCastMapper *volumeMapper = vtkVolumeRayCastMapper::New();
  volumeMapper->SetVolumeRayCastFunction( vtkVolumeRayCastCompositeFunction::New());
  volumeMapper->SetInputConnection( icast->GetOutputPort() );

  vtkVolume *volume = vtkVolume::New();
  volume->SetMapper( volumeMapper ); 
  volume->SetProperty( volumeProperty );

  vtkRenderer *renDicom = vtkRenderer::New();
  renDicom->AddViewProp( volume );

  return renDicom;
}
