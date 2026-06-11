
/*!
 ***********************************************************************
 *  \file
 *     decoder_test.c
 *  \brief
 *     H.264/AVC decoder test 
 *  \author
 *     Main contributors (see contributors.h for copyright, address and affiliation details)
 *     - Yuwen He       <yhe@dolby.com>
 ***********************************************************************
 */


#include "contributors.h"

#include <sys/stat.h>
#include <omp.h>

//#include "global.h"
#include "win32.h"
#include "h264decoder.h"
#include "configfile.h"
#ifdef BUILD_LDECOD_LIBRARY
#include "ldecod_api.h"
#endif // BUILD_LDECOD_LIBRARY

#define DECOUTPUT_TEST      0

#define PRINT_OUTPUT_POC    0
#define BITSTREAM_FILENAME  "test.264"
#define DECRECON_FILENAME   "test_dec.yuv"
#define ENCRECON_FILENAME   "test_rec.yuv"
#define FCFR_DEBUG_FILENAME "fcfr_dec_rpu_stats.txt"
#define DECOUTPUT_VIEW0_FILENAME  "H264_Decoder_Output_View0.yuv"
#define DECOUTPUT_VIEW1_FILENAME  "H264_Decoder_Output_View1.yuv"

#ifdef BUILD_LDECOD_LIBRARY
static void Configure(InputParameters *p_Inp, const ldecod_config_t *cfg)
#else
static void Configure(InputParameters *p_Inp, int ac, char *av[])
#endif
{
  // omp setup
  init_time();
  omp_set_dynamic(0);
  omp_set_num_threads(omp_get_num_procs());
  /* Make idle workers sleep, not spin */
#ifdef _WIN32
  _putenv_s("OMP_WAIT_POLICY", "PASSIVE"); /* Windows */
#endif
  //char *config_filename=NULL;
  //char errortext[ET_SIZE];
  memset(p_Inp, 0, sizeof(InputParameters));
  strcpy(p_Inp->infile, BITSTREAM_FILENAME); //! set default bitstream name
  strcpy(p_Inp->outfile, DECRECON_FILENAME); //! set default output file name
  strcpy(p_Inp->reffile, ENCRECON_FILENAME); //! set default reference file name
  
#ifdef _LEAKYBUCKET_
  strcpy(p_Inp->LeakyBucketParamFile,"leakybucketparam.cfg");    // file where Leaky Bucket parameters (computed by encoder) are stored
#endif
#ifdef BUILD_LDECOD_LIBRARY
    ParseCommand(p_Inp, 0, NULL);
#else
    ParseCommand(p_Inp, ac, av);
#endif

#ifdef BUILD_LDECOD_LIBRARY
  // ##########################################################################################
  // # Files
  // ##########################################################################################
  strcpy_s(p_Inp->infile,  FILE_NAME_SIZE, "ldecod-callback://");
  strcpy_s(p_Inp->outfile, FILE_NAME_SIZE, "ldecod-output://");
  // RefFile               = "test_rec.yuv"   # Ref sequence (for SNR)
  // WriteUV               = 1                # Write 4:2:0 chroma components for monochrome streams
  p_Inp->FileFormat = cfg->bitstream_format;
  // RefOffset             = 0                # SNR computation offset
  // POCScale              = 2                # Poc Scale (1 or 2)
  // ##########################################################################################
  // # HRD parameters
  // ##########################################################################################
  // #R_decoder             = 500000           # Rate_Decoder
  // #B_decoder             = 104000           # B_decoder
  // #F_decoder             = 73000            # F_decoder
  // #LeakyBucketParamFile  = "leakybucketparam.cfg" # LeakyBucket Params
  // ##########################################################################################
  // # decoder control parameters
  // ##########################################################################################
  // DisplayDecParams       = 0                # 1: Display parameters; 
  p_Inp->conceal_mode = cfg->concealment_mode;
  // RefPOCGap              = 2                # Reference POC gap (2: IPP (Default), 4: IbP / IpP)
  // POCGap                 = 2                # POC gap (2: IPP /IbP/IpP (Default), 4: IPP with frame skip = 1 etc.)
  p_Inp->silent = cfg->verbose ? 0 : 1;
  // IntraProfileDeblocking = 1                # Enable Deblocking filter in intra only profiles (0=disable, 1=filter according to SPS parameters)
  // DecFrmNum              = 0                # Number of frames to be decoded (-n)
  // ##########################################################################################
  // # MVC decoding parameters
  // ##########################################################################################
  p_Inp->DecodeAllLayers = 1;

  // ##########################################################################################
  // # Other parameters
  // ##########################################################################################  
  //p_Inp->source.yuv_format = (ColorFormat)cfg->output_format_hint;
  //p_Inp->output.yuv_format = (ColorFormat)cfg->output_format_hint;

  //p_Inp->dpb_plus[0] = cfg->dpb_size_override;
  //p_Inp->dpb_plus[1] = cfg->dpb_size_override;
#endif

  fprintf(stdout,"----------------------------- JM %s %s -----------------------------\n", VERSION, EXT_VERSION);
  //fprintf(stdout," Decoder config file                    : %s \n",config_filename);
  if(!p_Inp->bDisplayDecParams)
  {
    fprintf(stdout,"--------------------------------------------------------------------------\n");
    fprintf(stdout," Input H.264 bitstream                  : %s \n",p_Inp->infile);
    fprintf(stdout," Output decoded YUV                     : %s \n",p_Inp->outfile);
    //fprintf(stdout," Output status file                     : %s \n",LOGFILE);
    fprintf(stdout," Input reference file                   : %s \n",p_Inp->reffile);

    fprintf(stdout,"--------------------------------------------------------------------------\n");
  #ifdef _LEAKYBUCKET_
    fprintf(stdout," Rate_decoder        : %8ld \n",p_Inp->R_decoder);
    fprintf(stdout," B_decoder           : %8ld \n",p_Inp->B_decoder);
    fprintf(stdout," F_decoder           : %8ld \n",p_Inp->F_decoder);
    fprintf(stdout," LeakyBucketParamFile: %s \n",p_Inp->LeakyBucketParamFile); // Leaky Bucket Param file
    calc_buffer(p_Inp);
    fprintf(stdout,"--------------------------------------------------------------------------\n");
  #endif
  }
  
}

/*********************************************************
if bOutputAllFrames is 1, then output all valid frames to file onetime; 
else output the first valid frame and move the buffer to the end of list;
*********************************************************/
static int WriteOneFrame(DecodedPicList *pDecPic, int hFileOutput0, int hFileOutput1, int bOutputAllFrames)
{
  int iOutputFrame=0;
  DecodedPicList *pPic = pDecPic;

  if(pPic && (((pPic->iYUVStorageFormat==2) && pPic->bValid==3) || ((pPic->iYUVStorageFormat!=2) && pPic->bValid==1)) )
  {
    int i, iWidth, iHeight, iStride, iWidthUV, iHeightUV, iStrideUV;
    byte *pbBuf;    
    int hFileOutput;
    size_t res;

    iWidth = pPic->iWidth*((pPic->iBitDepth+7)>>3);
    iHeight = pPic->iHeight;
    iStride = pPic->iYBufStride;
    if(pPic->iYUVFormat != YUV444)
      iWidthUV = pPic->iWidth>>1;
    else
      iWidthUV = pPic->iWidth;
    if(pPic->iYUVFormat == YUV420)
      iHeightUV = pPic->iHeight>>1;
    else
      iHeightUV = pPic->iHeight;
    iWidthUV *= ((pPic->iBitDepth+7)>>3);
    iStrideUV = pPic->iUVBufStride;
    
    do
    {
      if(pPic->iYUVStorageFormat==2)
        hFileOutput = (pPic->iViewId&0xffff)? hFileOutput1 : hFileOutput0;
      else
        hFileOutput = hFileOutput0;
      if(hFileOutput >=0)
      {
        //Y;
        pbBuf = pPic->pY;
        for(i=0; i<iHeight; i++)
        {
          res = write(hFileOutput, pbBuf+i*iStride, iWidth);
          if (-1==res)
          {
            error ("error writing to output file.", 600);
          }
        }

        if(pPic->iYUVFormat != YUV400)
        {
         //U;
         pbBuf = pPic->pU;
         for(i=0; i<iHeightUV; i++)
         {
           res = write(hFileOutput, pbBuf+i*iStrideUV, iWidthUV);
           if (-1==res)
           {
             error ("error writing to output file.", 600);
           }
}
         //V;
         pbBuf = pPic->pV;
         for(i=0; i<iHeightUV; i++)
         {
           res = write(hFileOutput, pbBuf+i*iStrideUV, iWidthUV);
           if (-1==res)
           {
             error ("error writing to output file.", 600);
           }
         }
        }

        iOutputFrame++;
      }

      if (pPic->iYUVStorageFormat == 2)
      {
        hFileOutput = ((pPic->iViewId>>16)&0xffff)? hFileOutput1 : hFileOutput0;
        if(hFileOutput>=0)
        {
          int iPicSize =iHeight*iStride;
          //Y;
          pbBuf = pPic->pY+iPicSize;
          for(i=0; i<iHeight; i++)
          {
            res = write(hFileOutput, pbBuf+i*iStride, iWidth);
            if (-1==res)
            {
              error ("error writing to output file.", 600);
            }
          }

          if(pPic->iYUVFormat != YUV400)
          {
           iPicSize = iHeightUV*iStrideUV;
           //U;
           pbBuf = pPic->pU+iPicSize;
           for(i=0; i<iHeightUV; i++)
           {
             res = write(hFileOutput, pbBuf+i*iStrideUV, iWidthUV);
             if (-1==res)
             {
               error ("error writing to output file.", 600);
             }
           }
           //V;
           pbBuf = pPic->pV+iPicSize;
           for(i=0; i<iHeightUV; i++)
           {
             res = write(hFileOutput, pbBuf+i*iStrideUV, iWidthUV);
             if (-1==res)
             {
               error ("error writing to output file.", 600);
             }
           }
          }

          iOutputFrame++;
        }
      }

#if PRINT_OUTPUT_POC
      fprintf(stdout, "\nOutput frame: %d/%d\n", pPic->iPOC, pPic->iViewId);
#endif
      pPic->bValid = 0;
      pPic = pPic->pNext;
    }while(pPic != NULL && pPic->bValid && bOutputAllFrames);
  }
#if PRINT_OUTPUT_POC
  else
    fprintf(stdout, "\nNone frame output\n");
#endif

  return iOutputFrame;
}

/*!
 ***********************************************************************
 * \brief
 *    main function for JM decoder
 ***********************************************************************
 */
#ifdef BUILD_LDECOD_LIBRARY
int jm_ldecod_run_with_config(const ldecod_config_t *cfg)
#else
int main(int argc, char **argv)
#endif
{
  int iRet;
  DecodedPicList *pDecPicList;
  int hFileDecOutput0=-1, hFileDecOutput1=-1;
  int iFramesOutput=0, iFramesDecoded=0;
  InputParameters InputParams;

#if DECOUTPUT_TEST
  hFileDecOutput0 = open(DECOUTPUT_VIEW0_FILENAME, OPENFLAGS_WRITE, OPEN_PERMISSIONS);
  fprintf(stdout, "Decoder output view0: %s\n", DECOUTPUT_VIEW0_FILENAME);
  hFileDecOutput1 = open(DECOUTPUT_VIEW1_FILENAME, OPENFLAGS_WRITE, OPEN_PERMISSIONS);
  fprintf(stdout, "Decoder output view1: %s\n", DECOUTPUT_VIEW1_FILENAME);
#endif

  init_time();

  //get input parameters;
#ifdef BUILD_LDECOD_LIBRARY
  Configure(&InputParams, cfg);
#else
  Configure(&InputParams, argc, argv);
#endif

  //open decoder;
  iRet = OpenDecoder(&InputParams);
  if(iRet != DEC_OPEN_NOERR)
  {
    fprintf(stderr, "Open encoder failed: 0x%x!\n", iRet);
    return -1; //failed;
  }

  //decoding;
  do
  {
    iRet = DecodeOneFrame(&pDecPicList);
    if(iRet==DEC_EOS || iRet==DEC_SUCCEED)
    {
      //process the decoded picture, output or display;
      iFramesOutput += WriteOneFrame(pDecPicList, hFileDecOutput0, hFileDecOutput1, 0);
      iFramesDecoded++;
    }
    else
    {
      //error handling;
      fprintf(stderr, "Error in decoding process: 0x%x\n", iRet);
    }
  }while((iRet == DEC_SUCCEED) && ((p_Dec->p_Inp->iDecFrmNum==0) || (iFramesDecoded<p_Dec->p_Inp->iDecFrmNum)));

  iRet = FinitDecoder(&pDecPicList);
  iFramesOutput += WriteOneFrame(pDecPicList, hFileDecOutput0, hFileDecOutput1 , 1);
  iRet = CloseDecoder();

  //quit;
  if(hFileDecOutput0>=0)
  {
    close(hFileDecOutput0);
  }
  if(hFileDecOutput1>=0)
  {
    close(hFileDecOutput1);
  }

  printf("%d frames are decoded.\n", iFramesDecoded);
  return 0;
}

/*
 * Externally-triggered early termination for an active decode.
 *
 * Reads the process-global decoder pointer p_Dec set up by OpenDecoder;
 * if no decoder is active (p_Dec == NULL), returns without doing
 * anything. Otherwise calls jm_demux_stop on the decoder's video
 * context, which closes the internal NALU demux queue. The decode
 * thread's blocking jm_nalu_queue_pop then returns end-of-stream and
 * the surrounding ldecod_decode() unwinds normally.
 *
 * Callable from any thread other than the one currently inside
 * ldecod_decode(). Idempotent: stopping an already-closed queue is a
 * no-op on the JM side. Public API entry point is
 * ldecod_api_stop_decoding() in ldecod_api.h; the wrapper exists so
 * the public namespace stays ldecod_api_*.
 */
void ldecod_stop_decoding(void)
{
  DecoderParams *pDecoder = p_Dec;
  if (!pDecoder)
    return;
  jm_demux_stop(pDecoder->p_Vid);
}
