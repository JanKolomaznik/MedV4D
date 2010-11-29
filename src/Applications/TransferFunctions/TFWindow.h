#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QMainWindow>
#include <QtGui/QDockWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QMenuBar>
#include <QtGui/QKeySequence>

#include <TFHolderFactory.h>
#include <TFPaletteButton.h>
#include <TFDockHolder.h>
#include <ui_TFWindow.h>

namespace M4D {
namespace GUI {

class TFWindow : public QWidget{

    Q_OBJECT

	typedef std::map<TFSize, TFDockHolder*> DockHolderMap;
	typedef DockHolderMap::iterator DockHolderMapIt;

	typedef std::map<TFSize, TFAbstractHolder*> HolderMap;
	typedef HolderMap::iterator HolderMapIt;

public:

	TFWindow(QMainWindow* parent);
    ~TFWindow();

	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		if(activeHolder_ < 0) return false;
		return palette_.find(activeHolder_)->second->applyTransferFunction<ElementIterator>(begin, end);
	}

	void createMenu(QMenuBar* menubar);

	void setupDefault();

signals:

	void ResizeHolder(const TFSize& index, const QRect& rect);

protected slots:

    void close_triggered();
    void save_triggered();
    void load_triggered();
	void newTF_triggered(const TFHolderType& tfType);

	void change_activeHolder(const TFSize& index);
	void release_triggered();

protected:

	void mousePressEvent(QMouseEvent*);
	void keyPressEvent(QKeyEvent*);
	void resizeEvent(QResizeEvent*);

private:	

	class Indexer{

		typedef std::vector<TFSize> Indexes;
		typedef Indexes::iterator IndexesIt;

	public:

		Indexer();
		~Indexer();

		TFSize getIndex();
		void releaseIndex(const TFSize& index);

	private:

		TFSize nextIndex_;
		Indexes released_;
	};

    Ui::TFWindow* ui_;
	QMainWindow* mainWindow_;

	QMenu* menuTF_;
	QMenu* menuNew_;
    QAction *actionLoad_;
    QAction *actionSave_;
    QAction *actionExit_;

	Indexer indexer_;
	int activeHolder_;
	int holderInWindow_;
	HolderMap palette_;
	DockHolderMap releasedHolders_;
	HolderMap inWindowHolders_;
	
	TFActions tfActions_;	

	void addToPalette_(TFAbstractHolder* holder);
	void removeFromPalette_();

	TFSize getFirstInWindow_();
	void changeHolderInWindow_(const TFSize& index, const bool& hideOld);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW