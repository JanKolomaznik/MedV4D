#ifndef VIEWER_DESKTOP_H
#define VIEWER_DESKTOP_H


class ViewerDesktop: public QWidget
{
	Q_OBJECT;
public:

public slots:
	void
	setLayoutOrganization( int cols, int rows );

protected:
	AGLViewer *
	createViewer();
	
	struct ViewerInfo
	{
		AGLViewer *viewer;
	};

	typedef std::vector< ViewerInfo > ViewerList;

	ViewerList mViewers;

}



#endif /*VIEWER_DESKTOP_H*/
