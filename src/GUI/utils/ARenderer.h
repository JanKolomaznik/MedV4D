#ifndef ARENDERER_H
#define ARENDERER_H


class ARenderer
{
public:
	virtual void
	Initialize() = 0;

	virtual void
	Finalize() = 0;

	virtual void
	Render() = 0;

};

#endif /*ARENDERER_H*/
