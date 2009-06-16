#ifndef SUPPORT_H_
#define SUPPORT_H_

typedef union {
	float32 f;
	uint32 i;
} UFloatToInt;

static float32 INT_TO_FLOAT(uint32 x)
{
	UFloatToInt uConv; uConv.i = x;
	return uConv.f;
}

static uint32_t FLOAT_TO_INT(float32 f)
{
	UFloatToInt uConv; uConv.f = f;
	return uConv.i;
}

#endif /*SUPPORT_H_*/
