#ifndef TFACTION
#define TFACTION

#include <TFTypes.h>
#include <QtGui/QAction>

namespace M4D {
namespace GUI {

class TFAction: public QAction{

	Q_OBJECT

public:
	TFAction(QObject* parent, TFType tfType);

	~TFAction();

signals:
	void TFActionClicked(TFType &tfType);

public slots:
	void action_triggered();

private:	
	TFType type_;
};

typedef std::vector<TFAction*> TFActions;
typedef std::vector<TFAction*>::iterator TFActionsIt;

} // namespace GUI
} // namespace M4D

#endif	//TFACTION