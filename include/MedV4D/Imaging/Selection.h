#ifndef _SELECTION_H
#define _SELECTION_H


/**
 * Base class for storing selection of geometry objects.
 **/
class SelectionBase
{
public:

        virtual void
        Reset();
};

/**
 * Base class for storing selection of subelements of geometry objects.
 **/
class SubSelectionBase
{
public:

        virtual void
        Reset();
};


#endif /*_SELECTION_H*/
