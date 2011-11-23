#ifndef PLOT_BOX_H
#define PLOT_BOX_H

#include <QMap>
#include <QPixmap>
#include <QVector>
#include <QWidget>

#include "SettingsBox.h"


class QToolButton;
class PlotSettings;

/**
 * Plotting widget - based on "Creating Custom Widgets" article from "C++ GUI Programming with Qt4"
 * Displays one or more curves specified as vectors of coordinates.
 * The user can draw a rubber band on the image, and the PlotBox will zoom in on the area enclosed
 * by the rubber band. The user draws the rubber band by clicking a point on the graph, dragging the
 * mouse to another position with the left mouse button held down, and releasing the mouse button.
 * The user can zoom in repeatedly by drawing a rubber band multiple times, zooming out using the 
 * Zoom Out button and then zooming back in using the Zoom In button. The Zoom In and Zoom Out buttons
 * appear the first time they become available so that they don't clutter the display if the user doesn't
 * zoom the graph.
 * The widget can hold the data for up to CURVE_NUMBER curves - if the Hold graps is checked. It also 
 * maintains a stack of PlotSettings objects, each of which corresponds to a particular zoom level.
 */
class PlotBox: public QWidget
{
	Q_OBJECT

  public:

    /// Providing some spacing around the graph.
    static const int MARGIN = 28;

    /// Limiting the number of curves - how many to draw when the graph is held.
    static const unsigned CURVE_NUMBER = 6;

	  /** 
     * PlotBox constructor.
     * 
     * @param analysisFilter pointer to the analysis filter - to ask for the data of curves
     * @param parent pointer to the parent
     */
    PlotBox ( M4D::Imaging::APipeFilter *analysisFilter, QWidget *parent = 0 );

    /**
     * Specifies the PlotSettings to use for displaying the plot.
     * 
     * @param settings reference to the initial PlotSettings
     */
    void setPlotSettings ( const PlotSettings &settings );
    
    /**
     * Sets the curve data for a given curve ID.
     *
     * @param id ID of the given curve
     * @param data reference to the curve data
     */
    void setCurveData ( int id, const QVector< QPointF > &data );
    
    /**
     * Removes the specified curve from the curve map.
     * 
     * @param id ID of the curve to be removed
     */
    void clearCurve ( int id );
    
    /**
     * Specifies a widget's ideal minimum size.
     */
    QSize minimumSizeHint () const;
    
    /**
     * Specifies a widget's ideal size.
     */  
    QSize sizeHint () const;

  public slots:

    /**
     * Slot zooms out if the graph is zoomed in.
     */
    void zoomIn ();

    /**
     * Slot zooms in.
     */
    void zoomOut ();

    /**
     * Slot for mouse click handling (in dataset) - a curve need to be drawn.
     *
     * @param index index of the viewer, where the click occured
     * @param x x coordinate of the click (converted to the coordinates of the dataset)
     * @param y y coordinate of the click (converted to the coordinates of the dataset) 
     * @param z z coordinate of the click (converted to the coordinates of the dataset)
     */
    void pointPicked ( unsigned index, int x, int y, int z );

  protected:

	  /**
     * Performs all the drawing - by copying the pixmap onto the widget at position (0, 0).
     *
     * @param event not used
     */
    void paintEvent ( QPaintEvent *event );
    
    /**
     * Reimplementation of resizeEvent.
     *
     * @param event not used
     */
    void resizeEvent ( QResizeEvent *event );
    
    /**
     * Handles left button press - start displaying the rubber band.
     *
     * @param event occured event
     */
    void mousePressEvent ( QMouseEvent *event );
    
    /**
     * Handles dragging.
     *
     * @param event occured event
     */
    void mouseMoveEvent ( QMouseEvent *event );
    
    /**
     * Handles left button release - zoom.
     *
     * @param event occured event
     */   
    void mouseReleaseEvent ( QMouseEvent *event );
   
    /**
     * Handles key presses.
     *
     * @param event occured event
     */ 
    void keyPressEvent ( QKeyEvent *event );
    
    /**
     * Handles mouse wheel turn.
     *
     * @param event occured event
     */ 
    void wheelEvent ( QWheelEvent *event );

  private:

    /**
     * Erases or redraws the rubber band.
     */
    void updateRubberBandRegion ();
    
    /**
     * Redraws the plot onto the off-screen pixmap and updates the display.
     */
    void refreshPixmap ();
    
    /**
     * Draws the grid behind the curves and the axes.
     *
     * @param painter pointer to the painter
     */
    void drawGrid ( QPainter *painter );
    
    /**
     * Draws the curves on top of the grid.
     *
     * @param painter pointer to the painter
     */
    void drawCurves ( QPainter *painter );

    /// Pointer to the analysis filter.
    M4D::Imaging::APipeFilter *analysisFilter;

    /// Buttons for zooming in, out, holding graph.
    QToolButton *zoomInButton, *zoomOutButton, *holdButton;

    /// Storage for curves (curve: QVector< QPointF >, where QPointF is a floating-point version of QPoint)
    QMap< int, QVector< QPointF > > curveMap;
    
    /// Stack for the different zoom settings.
    QVector< PlotSettings > zoomStack;
    
    /// Current PlotSettings's index in the zoomStack.
    int curZoom;
    
    /// Flag for the rubber band manipulation.   
    bool rubberBandIsShown;
    
    /// Rectangle representing the rubber band.
    QRect rubberBandRect;
    
    /// Pixmap for holding a copy of the whole widget's rendering - the plot is always drawn onto this
    /// off-screen pixmap first
    QPixmap pixmap;

    /// Number of drawn curves
    unsigned curveCount;
};


/**
 * The PlotSettings class specifies the range of the x- and y-axes and the number of ticks for these axes. 
 */
class PlotSettings
{
  public:

    /// Axes limits.
    static const unsigned MIN_X = 0;
    static const unsigned MAX_X = 42;
    static const unsigned MIN_Y = 1020;
    static const unsigned MAX_Y = 1230;
    
    /** 
     * PlotSettings constructor.
     */
    PlotSettings ();

    /** 
     * Increments (or decrements) minX, maxX, minY, and maxY by the interval between two ticks times a given number.
     * 
     * @param dx value of x increment/decrement
     * @param dx value of y increment/decrement
     */
    void scroll ( int dx, int dy );
     
    /** 
     * Rounds the minX, maxX, minY, and maxY values to "nice" values and determines the number 
     * of ticks appropriate for each axis
     */
    void adjust ();

    /** 
     * Returns x range.
     */
    double spanX () const { return maxX - minX; }

    /** 
     * Returns y range.
     */
    double spanY () const { return maxY - minY; }

    /// Ranges for each axis.
    double minX, maxX, minY, maxY;

    /// Number of ticks for each axis.
    int numXTicks, numYTicks;

  private:

    /** 
     * Converts its min and max parameters into "nice" numbers and sets its numTicks parameter to the number of 
     * ticks it calculates to be appropriate for the given [min, max] range.
     * 
     * @param min reference to the min value to be adjusted
     * @param max reference to the max value to be adjusted
     * @param numTicks reference to the numTicks value to be adjusted
     */
    static void adjustAxis ( double &min, double &max, int &numTicks );
};

#endif // PLOT_BOX_H


