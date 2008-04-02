#include "renderWindowUtils.h"

vtkRenderer *sphereToRenderWindow ()
{
  // just for testing, won't be in real application
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


vtkRenderer *dicomToRenderWindow ()
{
  // just for testing, won't be in real application
  vtkDICOMImageReader *reader = vtkDICOMImageReader::New();
  reader->SetDirectoryName( "..\\..\\GUI\\qtVtk\\DICOM" ); 
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
  // renDicom->SetBackground( 1, 1, 1 );	
  renDicom->AddViewProp( volume );

  return renDicom;
}