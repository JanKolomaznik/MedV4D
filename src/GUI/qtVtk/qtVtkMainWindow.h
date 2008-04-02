#ifndef QT_VTK_MAIN_WINDOW_H
#define QT_VTK_MAIN_WINDOW_H

#include "ui_QtVtkMainWindow.h"

#include "renderWindowUtils.h"


class MainWindow: public QMainWindow
{
  Q_OBJECT

  public:
    MainWindow( QMainWindow *parent = 0 );

  private slots:
   // auto-connection naming convention required by uic
    void on_SphereRadio_clicked ();
    void on_DicomRadio_clicked ();

  private:
    Ui::MainWindow ui;

  private:
    vtkRenderWindow *renWin;
    vtkRenderWindowInteractor *iren;
    vtkRenderer *renSphere, *renDicom;
};

#endif // QT_VTK_MAIN_WINDOW_H