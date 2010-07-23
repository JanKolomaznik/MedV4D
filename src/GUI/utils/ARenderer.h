#ifndef ARENDERER_H
#define ARENDERER_H

namespace M4D
{
	
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

} /*namespace M4D*/

#endif /*ARENDERER_H*/
