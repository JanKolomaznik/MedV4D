#ifndef TF_ACTION
#define TF_ACTION

#include <TFTypes.h>
#include <TFHolder.h>
#include <QtGui/QAction>

namespace M4D {
namespace GUI {

class TFAction: public QAction{

	Q_OBJECT

public:
	TFAction(QObject* parent, TFHolder::Type tfType);

	~TFAction();

signals:
	void TFActionClicked(const TFHolder::Type &tfType);

public slots:
	void action_triggered();

private:	
	TFHolder::Type type_;
};

typedef std::vector<TFAction*> TFActions;
typedef std::vector<TFAction*>::iterator TFActionsIt;

} // namespace GUI
} // namespace M4D

#endif	//TF_ACTION