/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file ViewerFactory.h
 * @{ 
 **/

#ifndef VIEWER_FACTORY_H
#define VIEWER_FACTORY_H

#include "GUI/widgets/m4dGUIAbstractViewerWidget.h"


namespace M4D {
namespace GUI {

#define SLICE_VIEWER_ID   0
#define VTK_VIEWER_ID     1

class ViewerFactory
{ 
  public:

    virtual M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( M4D::Imaging::ConnectionInterface* conn,
                                                                 unsigned index ) = 0;
    virtual M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( unsigned index ) = 0;
    
    virtual unsigned getID () const = 0;
    virtual const char *getDescription () const = 0;
};


class SliceViewerFactory: public ViewerFactory
{
  public:

    M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( M4D::Imaging::ConnectionInterface* conn,
                                                         unsigned index );
    M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( unsigned index );

    unsigned getID () const { return ID; };
    const char *getDescription () const { return description; };

  private:

    static const unsigned ID; 
    static const char *description;
};


class VtkViewerFactory: public ViewerFactory
{
  public:

    M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( M4D::Imaging::ConnectionInterface* conn,
                                                         unsigned index ); 
    M4D::Viewer::m4dGUIAbstractViewerWidget *newViewer ( unsigned index );

    unsigned getID () const { return ID; };
    const char *getDescription () const { return description; };

  private:

    static const unsigned ID; 
    static const char *description;
};

} // namespace GUI
} // namespace M4D

#endif // VIEWER_FACTORY_H


/** @} */

