#ifndef _TRANSFORM_SSE_H_
#define _TRANSFORM_SSE_H_

extern void inverse4x4_sse(int **tblock, int **block, int pos_y, int pos_x);
extern void inverse8x8_sse2(int **tblock, int **block, int pos_x);

#endif //_TRANSFORM_SSE_H_
