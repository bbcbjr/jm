
/*!
 ***************************************************************************
 * \file
 *    biaridecod.h
 *
 * \brief
 *    Headerfile for binary arithmetic decoder routines
 *
 * \author
 *    Detlev Marpe,
 *    Gabi Blättermann
 *    Copyright (C) 2000 HEINRICH HERTZ INSTITUTE All Rights Reserved.
 *
 * \date
 *    21. Oct 2000
 **************************************************************************
 */

#ifndef _BIARIDECOD_H_
#define _BIARIDECOD_H_

#include "jm_defines.h"

/************************************************************************
 * D e f i n i t i o n s
 ***********************************************************************
 */

/* Range table for  LPS */
static const byte rLPS_table_64x4[64][4]=
{
  { 128, 176, 208, 240},
  { 128, 167, 197, 227},
  { 128, 158, 187, 216},
  { 123, 150, 178, 205},
  { 116, 142, 169, 195},
  { 111, 135, 160, 185},
  { 105, 128, 152, 175},
  { 100, 122, 144, 166},
  {  95, 116, 137, 158},
  {  90, 110, 130, 150},
  {  85, 104, 123, 142},
  {  81,  99, 117, 135},
  {  77,  94, 111, 128},
  {  73,  89, 105, 122},
  {  69,  85, 100, 116},
  {  66,  80,  95, 110},
  {  62,  76,  90, 104},
  {  59,  72,  86,  99},
  {  56,  69,  81,  94},
  {  53,  65,  77,  89},
  {  51,  62,  73,  85},
  {  48,  59,  69,  80},
  {  46,  56,  66,  76},
  {  43,  53,  63,  72},
  {  41,  50,  59,  69},
  {  39,  48,  56,  65},
  {  37,  45,  54,  62},
  {  35,  43,  51,  59},
  {  33,  41,  48,  56},
  {  32,  39,  46,  53},
  {  30,  37,  43,  50},
  {  29,  35,  41,  48},
  {  27,  33,  39,  45},
  {  26,  31,  37,  43},
  {  24,  30,  35,  41},
  {  23,  28,  33,  39},
  {  22,  27,  32,  37},
  {  21,  26,  30,  35},
  {  20,  24,  29,  33},
  {  19,  23,  27,  31},
  {  18,  22,  26,  30},
  {  17,  21,  25,  28},
  {  16,  20,  23,  27},
  {  15,  19,  22,  25},
  {  14,  18,  21,  24},
  {  14,  17,  20,  23},
  {  13,  16,  19,  22},
  {  12,  15,  18,  21},
  {  12,  14,  17,  20},
  {  11,  14,  16,  19},
  {  11,  13,  15,  18},
  {  10,  12,  15,  17},
  {  10,  12,  14,  16},
  {   9,  11,  13,  15},
  {   9,  11,  12,  14},
  {   8,  10,  12,  14},
  {   8,   9,  11,  13},
  {   7,   9,  11,  12},
  {   7,   9,  10,  12},
  {   7,   8,  10,  11},
  {   6,   8,   9,  11},
  {   6,   7,   9,  10},
  {   6,   7,   8,   9},
  {   2,   2,   2,   2}
};


static const byte AC_next_state_MPS_64[64] =
{
  1,2,3,4,5,6,7,8,9,10,
  11,12,13,14,15,16,17,18,19,20,
  21,22,23,24,25,26,27,28,29,30,
  31,32,33,34,35,36,37,38,39,40,
  41,42,43,44,45,46,47,48,49,50,
  51,52,53,54,55,56,57,58,59,60,
  61,62,62,63
};


static const byte AC_next_state_LPS_64[64] =
{
  0, 0, 1, 2, 2, 4, 4, 5, 6, 7,
  8, 9, 9,11,11,12,13,13,15,15,
  16,16,18,18,19,19,21,21,22,22,
  23,24,24,25,26,26,27,27,28,29,
  29,30,30,30,31,32,32,33,33,33,
  34,34,35,35,35,36,36,36,37,37,
  37,38,38,63
};

static const byte renorm_table_32[32]={6,5,4,4,3,3,3,3,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

extern void arideco_start_decoding(DecodingEnvironmentPtr eep, unsigned char *code_buffer, int firstbyte, int *code_len);
extern int  arideco_bits_read(DecodingEnvironmentPtr dep);
extern void arideco_done_decoding(DecodingEnvironmentPtr dep);
extern void biari_init_context (int qp, BiContextTypePtr ctx, const char* ini);
extern unsigned int biari_decode_symbol_eq_prob(DecodingEnvironmentPtr dep);
extern unsigned int biari_decode_final(DecodingEnvironmentPtr dep);


/*!
 ************************************************************************
 * \brief
 *    read two bytes from the bitstream
 ************************************************************************
 */
JM_FORCEINLINE unsigned int getword(DecodingEnvironmentPtr dep) {
  int *len = dep->Dcodestrm_len;
  byte *p_code_strm = &dep->Dcodestrm[*len];
#if (TRACE == 2)
  fprintf(p_trace, "get_byte: %d\n", *len);
  fprintf(p_trace, "get_byte: %d\n", *len + 1);
#endif
  *len += 2;
  return ((*p_code_strm << 8) | *(p_code_strm + 1));
}

/*!
************************************************************************
* \brief
*    biari_decode_symbol():
* \return
*    the decoded symbol
************************************************************************
*/
/*!
************************************************************************
* \brief
*    biari_decode_symbol():
* \return
*    the decoded symbol
************************************************************************
*/
JM_FORCEINLINE unsigned int biari_decode_symbol(DecodingEnvironment * restrict dep,
                                                BiContextType * restrict bi_ct)
{
  unsigned int range     = dep->Drange;
  unsigned int value     = dep->Dvalue;
  int          DbitsLeft = dep->DbitsLeft;
  unsigned int state     = bi_ct->state;
  unsigned int MPS       = bi_ct->MPS;

  unsigned int rLPS        = rLPS_table_64x4[state][(range >> 6) & 0x03];
  range                   -= rLPS;
  unsigned int scaledRange = range << DbitsLeft;   // compute once, reuse in LPS path
  unsigned int bit;

  if (value < scaledRange)
  {
    bit   = MPS;
    state = AC_next_state_MPS_64[state];
    if (range < 0x0100)
    {
      range    <<= 1;
      DbitsLeft -= 1;
    }
  }
  else                                // LPS
  {
    int renorm = renorm_table_32[rLPS >> 3];   // & 0x1F is a no-op (rLPS <= 240)
    value     -= scaledRange;                  // reuse the value we already computed
    range      = rLPS << renorm;
    DbitsLeft -= renorm;
    bit        = MPS ^ 0x01;
    MPS ^= (state == 0);
    state      = AC_next_state_LPS_64[state];
  }

  if (DbitsLeft <= 0)
  {
    value     = (value << 16) | getword(dep);
    DbitsLeft += 16;
  }

  // Single write-back at the end
  dep->Drange    = range;
  dep->Dvalue    = value;
  dep->DbitsLeft = DbitsLeft;
  bi_ct->state   = (uint16)state;
  bi_ct->MPS     = (byte)MPS;
  return bit;
}

#endif  // BIARIDECOD_H_

