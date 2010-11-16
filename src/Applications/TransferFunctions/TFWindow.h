#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QMenuBar>
#include <QtGui/QKeySequence>

#include <TFHolderFactory.h>
#include <TFPaletteButton.h>
#include <ui_TFWindow.h>

namespace M4D {
namespace GUI {

class TFWindow : public QWidget{

    Q_OBJECT

public:
	TFWindow();
    ~TFWindow();

	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		if(activeHolder_ < 0) return false;
		return palette_[activeHolder_]->applyTransferFunction<ElementIterator>(begin, end);
	}

	void createMenu(QMenuBar* menubar);

	void setupDefault();

signals:
	void ResizeHolder(const QRect& rect);

protected slots:
    void close_triggered();
    void save_triggered();
    void load_triggered();
	void newTF_triggered(const TFHolderType& tfType);

	void change_holder(const TFSize& index, const bool& forceChange = false);

protected:
	void resizeEvent(QResizeEvent* e);

private:	
    Ui::TFWindow* ui_;

	QMenu* menuTF_;
	QMenu* menuNew_;
    QAction *actionLoad_;
    QAction *actionSave_;
    QAction *actionExit_;

	int activeHolder_;
	TFPalette palette_;
	TFActions tfActions_;

	void addToPalette_(TFAbstractHolder* holder);
	void removeActiveFromPalette_();
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW