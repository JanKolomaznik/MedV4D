#include "GUI/m4dGUIVtkViewerWidget.h"

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
// just for imageDataToRenderWindow ( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects )
#include "Imaging/ImageFactory.h"

using namespace M4D::vtkIntegration;


m4dGUIVtkViewerWidget::m4dGUIVtkViewerWidget ( QVTKWidget *parent )
  : QVTKWidget( parent )
{}


void m4dGUIVtkViewerWidget::addRenderer ( vtkRenderer *renderer )
{ 
  vtkRenderWindow *rWin;
  rWin = GetRenderWindow();

  // remove previous one
  rWin->RemoveRenderer( rWin->GetRenderers()->GetFirstRenderer() );

  // add the actual
  rWin->AddRenderer( renderer );

  vtkRenderWindowInteractor *iren;
  iren = GetInteractor();
  iren->SetRenderWindow( rWin );
}


vtkRenderer *m4dGUIVtkViewerWidget::imageDataToRenderWindow ()
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
  //volumeProperty->ShadeOn(); 
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


// just for testing ImageFactory::CreateImageFromDICOM 
vtkRenderer *m4dGUIVtkViewerWidget::imageDataToRenderWindow ( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects )
{
  m4dImageDataSource *imageData = m4dImageDataSource::New();
  M4D::Imaging::AbstractImage::AImagePtr visualizedImage = 
				M4D::Imaging::ImageFactory::CreateImageFromDICOM( dicomObjects );

  D_COMMAND( if( !visualizedImage ){ D_PRINT( "INVALID IMAGE POINTER" );} );

  imageData->SetImageData( visualizedImage );

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
  //volumeProperty->ShadeOn(); 
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
vtkRenderer *m4dGUIVtkViewerWidget::sphereToRenderWindow ()
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
  renSphere->SetBackground( 0, 0, 0 );				// background color black

  return renSphere;
}


// just for testing, won't be in real application
vtkRenderer *m4dGUIVtkViewerWidget::dicomToRenderWindow ( const char *dirName )
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
