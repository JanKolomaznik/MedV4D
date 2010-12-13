#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QMainWindow>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QMenuBar>
#include <QtGui/QKeySequence>
#include <QtGui/QKeyEvent>

#include <TFHolderFactory.h>
#include <TFPaletteButton.h>
#include <TFDockHolder.h>
#include <ui_TFPalette.h>

namespace M4D {
namespace GUI {

class TFPalette : public QMainWindow{

    Q_OBJECT

	typedef std::map<TFSize, TFAbstractHolder*> HolderMap;
	typedef HolderMap::iterator HolderMapIt;

public:

	TFPalette(QMainWindow* parent);
    ~TFPalette();

	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		if(activeHolder_ < 0) return false;
		return palette_.find(activeHolder_)->second->applyTransferFunction<ElementIterator>(begin, end);
	}

	void setupDefault();
/*
signals:

	void ResizeHolder(TFSize index, QRect rect);
*/
protected slots:

    void close_triggered(TFSize index);
	void newTF_triggered(TFHolderType tfType);

    void on_actionLoad_triggered();

	void change_activeHolder(TFSize index);
	//void release_triggered();

protected:

	void resizeEvent(QResizeEvent*);

private:	

	class Indexer{

		typedef std::vector<TFSize> Indexes;
		typedef Indexes::iterator IndexesIt;

	public:

		Indexer();
		~Indexer();

		TFSize getIndex();
		void releaseIndex(TFSize index);

	private:

		TFSize nextIndex_;
		Indexes released_;
	};

    Ui::TFPalette* ui_;
	QMainWindow* mainWindow_;

	Indexer indexer_;
	int activeHolder_;
	HolderMap palette_;
	
	TFActions tfActions_;	

	bool connectTFActions_();

	void addToPalette_(TFAbstractHolder* holder);
	void removeFromPalette_(TFSize index);

	//TFSize getFirstInWindow_();
	//void changeHolderInWindow_(TFSize index, bool hideOld);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW