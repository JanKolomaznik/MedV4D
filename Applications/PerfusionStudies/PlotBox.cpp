#include "PlotBox.h"

#include "MainWindow.h"


PlotBox::PlotBox ( M4D::Imaging::APipeFilter *analysisFilter, QWidget *parent )
	: analysisFilter( analysisFilter ), QWidget ( parent )
{
  curveCount = 0;

  setBackgroundRole( QPalette::Dark );
  setAutoFillBackground( true );
  setSizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding );
  setFocusPolicy( Qt::StrongFocus );
  
  rubberBandIsShown = false;

  zoomInButton = new QToolButton( this );
  zoomInButton->setIcon( QIcon( ":/resources/zoomin.png" ) );
  zoomInButton->setToolTip( tr( "Zoom In" ) );
  zoomInButton->adjustSize();
 
  connect( zoomInButton, SIGNAL(clicked()), this, SLOT(zoomIn()) );

  zoomOutButton = new QToolButton( this );
  zoomOutButton->setIcon( QIcon( ":/resources/zoomout.png" ) );
  zoomOutButton->setToolTip( tr( "Zoom Out" ) );
  zoomOutButton->adjustSize();
  
  connect( zoomOutButton, SIGNAL(clicked()), this, SLOT(zoomOut()) );

  holdButton = new QToolButton( this );
  holdButton->setIcon( QIcon( ":/resources/hold.png" ) );
  holdButton->setToolTip( tr( "Hold graph" ) );
  holdButton->setCheckable( true );
  holdButton->adjustSize();
  
  connect( holdButton, SIGNAL(clicked()), this, SLOT(hold()) );

  setPlotSettings( PlotSettings() );
}


void PlotBox::setPlotSettings ( const PlotSettings &settings )
{
  zoomStack.clear();
  zoomStack.append( settings );
  
  curZoom = 0;
  
  zoomInButton->hide();
  zoomOutButton->hide();
  
  refreshPixmap();
}


void PlotBox::zoomIn ()
{
  if ( curZoom < zoomStack.count() - 1 ) 
  {
    ++curZoom;

    zoomInButton->setEnabled( curZoom < zoomStack.count() - 1 );
    zoomOutButton->setEnabled( true );
    zoomOutButton->show();

    refreshPixmap();
  }
}


void PlotBox::zoomOut ()
{
  if ( curZoom > 0 ) 
  {
    --curZoom;

    zoomOutButton->setEnabled( curZoom > 0 );
    zoomInButton->setEnabled( true );
    zoomInButton->show();

    refreshPixmap();
  }
}


void PlotBox::pointPicked ( unsigned index, int x, int y, int z )
{
  vector< ElementType > &curve = static_cast< Analysis * >( analysisFilter )->GetSmoothedCurve( x, y, z ); 

  QVector< QPointF > data;
  for ( unsigned i = 0; i < curve.size(); i++ ) {
    data.push_back( QPointF( i, curve[i] ) );
  }
  
  if ( holdButton->isChecked() ) {
    curveCount++;
  } 
  else 
  {
    curveCount = 0;
    curveMap.clear();
  }

  setCurveData( curveCount % CURVE_NUMBER, data );
}


void PlotBox::setCurveData ( int id, const QVector< QPointF > &data )
{
  curveMap[id] = data;

  refreshPixmap();
}


void PlotBox::clearCurve ( int id )
{
  curveMap.remove( id );

  refreshPixmap();
}


QSize PlotBox::minimumSizeHint () const
{
  return QSize( 6 * MARGIN, 4 * MARGIN );
}


QSize PlotBox::sizeHint () const
{
  return QSize( 12 * MARGIN, 8 * MARGIN );
}


void PlotBox::paintEvent ( QPaintEvent * /* event */ )
{
  QStylePainter painter( this );
  painter.drawPixmap( 0, 0, pixmap );

  if ( rubberBandIsShown ) 
  {
    painter.setPen( palette().light().color() );
    painter.drawRect( rubberBandRect.normalized().adjusted( 0, 0, -1, -1 ) );
  }

  if ( hasFocus() ) 
  {
    QStyleOptionFocusRect option;
    option.initFrom( this );
    option.backgroundColor = palette().dark().color();

    painter.drawPrimitive( QStyle::PE_FrameFocusRect, option );
  }
}


void PlotBox::resizeEvent ( QResizeEvent * /* event */ )
{
  int x = width() - (zoomInButton->width() + zoomOutButton->width() + holdButton->width() + 15);

  zoomInButton->move( x, 5 );
  zoomOutButton->move( x + zoomInButton->width() + 5, 5 );
  holdButton->move( x + zoomInButton->width() + zoomOutButton->width() + 10, 5 );
  
  refreshPixmap();
}


void PlotBox::mousePressEvent ( QMouseEvent *event )
{
  QRect rect( MARGIN, MARGIN, width() - 2 * MARGIN, height() - 2 * MARGIN );

  if ( event->button() == Qt::LeftButton ) 
  {
    if ( rect.contains( event->pos() ) ) 
    {
      rubberBandIsShown = true;

      rubberBandRect.setTopLeft( event->pos() );
      rubberBandRect.setBottomRight( event->pos() );
      updateRubberBandRegion();

      setCursor( Qt::CrossCursor );
    }
  }
}


void PlotBox::mouseMoveEvent ( QMouseEvent *event )
{
  if ( rubberBandIsShown ) 
  {
    updateRubberBandRegion();
    rubberBandRect.setBottomRight( event->pos() );
    updateRubberBandRegion();
  }
}


void PlotBox::mouseReleaseEvent ( QMouseEvent *event )
{
  if ( event->button() == Qt::LeftButton && rubberBandIsShown ) 
  {
    rubberBandIsShown = false;

    updateRubberBandRegion();
    
    unsetCursor();

    QRect rect = rubberBandRect.normalized();
    if ( rect.width() < 4 || rect.height() < 4 ) {
      return;
    }
    rect.translate( -MARGIN, -MARGIN );

    PlotSettings prevSettings = zoomStack[curZoom];
    PlotSettings settings;
    
    double dx = prevSettings.spanX() / (width() - 2 * MARGIN);
    double dy = prevSettings.spanY() / (height() - 2 * MARGIN);
    
    settings.minX = prevSettings.minX + dx * rect.left();
    settings.maxX = prevSettings.minX + dx * rect.right();
    settings.minY = prevSettings.maxY - dy * rect.bottom();
    settings.maxY = prevSettings.maxY - dy * rect.top();
    settings.adjust();

    zoomStack.resize( curZoom + 1 );
    zoomStack.append( settings );
    
    zoomIn();
  }
}


void PlotBox::keyPressEvent ( QKeyEvent *event )
{
  switch ( event->key() ) 
  {
    case Qt::Key_Plus:
      zoomIn();
      break;

    case Qt::Key_Minus:
      zoomOut();
      break;

    case Qt::Key_Left:
      zoomStack[curZoom].scroll( -1, 0 );
      refreshPixmap();
      break;

    case Qt::Key_Right:
      zoomStack[curZoom].scroll( +1, 0 );
      refreshPixmap();
      break;

    case Qt::Key_Down:
      zoomStack[curZoom].scroll( 0, -1 );
      refreshPixmap();
      break;

    case Qt::Key_Up:
      zoomStack[curZoom].scroll( 0, +1 );
      refreshPixmap();
      break;

    default:
      QWidget::keyPressEvent( event );
  }
}


void PlotBox::wheelEvent ( QWheelEvent *event )
{
  int numDegrees = event->delta() / 8;
  int numTicks = numDegrees / 15;

  if ( event->orientation() == Qt::Horizontal ) {
    zoomStack[curZoom].scroll( numTicks, 0 );
  } else {
    zoomStack[curZoom].scroll( 0, numTicks );
  }

  refreshPixmap();
}


void PlotBox::updateRubberBandRegion ()
{
  QRect rect = rubberBandRect.normalized();

  update( rect.left(), rect.top(), rect.width(), 1 );
  update( rect.left(), rect.top(), 1, rect.height() );
  update( rect.left(), rect.bottom(), rect.width(), 1 );
  update( rect.right(), rect.top(), 1, rect.height() );
}


void PlotBox::refreshPixmap ()
{
  pixmap = QPixmap( size() );
  pixmap.fill( this, 0, 0 );

  QPainter painter( &pixmap );
  painter.initFrom( this );

  drawGrid( &painter );
  drawCurves( &painter );

  update();
}


void PlotBox::drawGrid ( QPainter *painter )
{
  QRect rect( MARGIN + 10, MARGIN + 5, width() - 2 * MARGIN, height() - 2 * MARGIN );
  if ( !rect.isValid() ) {
    return;
  }

  PlotSettings settings = zoomStack[curZoom];

  QPen quiteDark = palette().dark().color().light();
  QPen light = palette().light().color();

  for ( int i = 0; i <= settings.numXTicks; ++i ) 
  {
     int x = rect.left() + (i * (rect.width() - 1) / settings.numXTicks);

     double label = settings.minX + (i * settings.spanX() / settings.numXTicks);

     painter->setPen( quiteDark );
     painter->drawLine( x, rect.top(), x, rect.bottom() );
     painter->setPen( light );
     painter->drawLine( x, rect.bottom(), x, rect.bottom() + 5 );
     painter->drawText( x - 50, rect.bottom() + 5, 100, 20, Qt::AlignHCenter | Qt::AlignTop, QString::number( label ) );
  }

  for ( int j = 0; j <= settings.numYTicks; ++j )
  {
    int y = rect.bottom() - (j * (rect.height() - 1) / settings.numYTicks);

    double label = settings.minY + (j * settings.spanY() / settings.numYTicks);

    painter->setPen( quiteDark );
    painter->drawLine( rect.left(), y, rect.right(), y );
    painter->setPen( light );
    painter->drawLine( rect.left() - 5, y, rect.left(), y );
    painter->drawText( rect.left() - MARGIN, y - 10, MARGIN - 5, 20, Qt::AlignRight | Qt::AlignVCenter, QString::number( label ) );
  }

  painter->drawRect( rect.adjusted( 0, 0, -1, -1 ) );
}


void PlotBox::drawCurves ( QPainter *painter )
{
  static const QColor colorForIds[6] = { Qt::blue, Qt::red, Qt::green, Qt::cyan, Qt::magenta, Qt::yellow };

  PlotSettings settings = zoomStack[curZoom];

  QRect rect( MARGIN + 10, MARGIN + 5, width() - 2 * MARGIN, height() - 2 * MARGIN );
  if ( !rect.isValid() ) {
    return;
  }

  painter->setClipRect( rect.adjusted( +1, +1, -1, -1 ) );

  QMapIterator< int, QVector< QPointF > > i( curveMap );

  while ( i.hasNext() ) 
  {
    i.next();

    int id = i.key();
    QVector< QPointF > data = i.value();
    QPolygonF polyline( data.count() );

    for ( int j = 0; j < data.count(); ++j ) 
    {
      double dx = data[j].x() - settings.minX;
      double dy = data[j].y() - settings.minY;

      double x = rect.left() + (dx * (rect.width() - 1) / settings.spanX());
      double y = rect.bottom() - (dy * (rect.height() - 1) / settings.spanY());
      
      polyline[j] = QPointF( x, y );
    }

    painter->setPen( colorForIds[(uint)id % 6] );
    painter->drawPolyline( polyline );
  }
}


PlotSettings::PlotSettings ()
{
  minX = MIN_X;
  maxX = MAX_X;
  numXTicks = 6;

  minY = MIN_Y;
  maxY = MAX_Y;
  numYTicks = 5;
}


void PlotSettings::scroll ( int dx, int dy )
{
  double stepX = spanX() / numXTicks;
  minX += dx * stepX;
  maxX += dx * stepX;

  double stepY = spanY() / numYTicks;
  minY += dy * stepY;
  maxY += dy * stepY;
}


void PlotSettings::adjust ()
{
  adjustAxis( minX, maxX, numXTicks );
  adjustAxis( minY, maxY, numYTicks );
}


void PlotSettings::adjustAxis ( double &min, double &max, int &numTicks )
{
  const int MinTicks = 4;
  double grossStep = (max - min) / MinTicks;
  double step = pow( 10.0, floor( log10( grossStep ) ) );

  if ( 5 * step < grossStep ) {
    step *= 5;
  } else if ( 2 * step < grossStep ) {
    step *= 2;
  }

  numTicks = (int)(ceil( max / step ) - floor( min / step ));
  if ( numTicks < MinTicks ) {
    numTicks = MinTicks;
  }

  min = floor( min / step ) * step;
  max = ceil( max / step ) * step;
}



