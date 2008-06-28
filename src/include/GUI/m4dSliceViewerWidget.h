#ifndef _M4DSLICEVIEWERWIDGET_H
#define _M4DSLICEVIEWERWIDGET_H

#include <QGLWidget>
#include "Imaging/Image.h"
#include "Common.h"
#include "ExceptionBase.h"

namespace M4D
{
namespace Viewer
{

class m4dSliceViewerWidget : public QGLWidget
{
    Q_OBJECT

public:
    m4dSliceViewerWidget( Imaging::Image<int16, 3>::Ptr img, QWidget *parent = 0);
    ~m4dSliceViewerWidget();

protected:
    void paintGL();
    void resizeGL(int winW, int winH);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void setSliceNum( size_t num );
    void zoomImage( int amount );
    void moveImage( QPoint diff );
    void adjustBrightness( int amount );
    void adjustContrast( int amount );

private:
    
    Imaging::Image<int16, 3>::Ptr _image;
    
    QPoint _lastPos;
    QPoint _offset;
    int _sliceNum;
    double _zoomRate;
};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
