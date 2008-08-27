/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file StManagerHProtocolsComp.h 
 * @{ 
 **/

#ifndef ST_MANAGER_H_PROTOCOLS_COMP_H
#define ST_MANAGER_H_PROTOCOLS_COMP_H

#include <QWidget>


namespace M4D {
namespace GUI {

/**
 * Class representing one of the components of Study Manager Widget.
 * It's providing Hanging Protocols funcionality - not needed yet.
 */
class StManagerHProtocolsComp: public QWidget
{
  Q_OBJECT

  public:

    /** 
     * Study Manager Hanging Protocols Component constructor.
     *
     * @param parent pointer to the parent widget - default is 0
     */
    StManagerHProtocolsComp ( QWidget *parent = 0 );
};

} // namespace GUI
} // namespace M4D

#endif // ST_MANAGER_H_PROTOCOLS_COMP_H


/** @} */

