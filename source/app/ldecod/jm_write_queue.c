
/*!
 ***************************************************************************
 * \file
 *    jm_write_queue.c
 *
 * \brief
 *    Implementation for jm_write_queue.h. See header for design
 *    rationale and lifecycle rules.
 ***************************************************************************
 */

#include <stdlib.h>
#include <string.h>

#include "global.h"           /* for VideoParameters */
#include "mbuffer.h"          /* for free_storable_picture (cleanup) */
#include "memalloc.h"         /* for no_mem_exit */
#include "jm_threads.h"
#include "jm_write_queue.h"

/* ===================================================================== */
/*                                  QUEUE                                  */
/* ===================================================================== */

struct jm_write_queue
{
  write_request_t *buf;
  int              cap;
  int              head;
  int              tail;
  int              count;
  int              closed;
  int              inflight;   /* 1 while writer is mid-drain on a popped request */
  jm_mutex_t       m;
  jm_cond_t        not_full;
  jm_cond_t        not_empty;
  jm_cond_t        idle;        /* signalled when count==0 && inflight==0 */
};

jm_write_queue_t *jm_write_queue_create(int capacity)
{
  jm_write_queue_t *q;

  if (capacity <= 0) return NULL;

  q = (jm_write_queue_t *)calloc(1, sizeof(*q));
  if (q == NULL) return NULL;

  q->buf = (write_request_t *)calloc((size_t)capacity, sizeof(write_request_t));
  if (q->buf == NULL) { free(q); return NULL; }

  q->cap      = capacity;
  q->head     = 0;
  q->tail     = 0;
  q->count    = 0;
  q->closed   = 0;
  q->inflight = 0;
  jm_mutex_init(&q->m);
  jm_cond_init (&q->not_full);
  jm_cond_init (&q->not_empty);
  jm_cond_init (&q->idle);
  return q;
}

/* Drop the writer's reference on a stranded request -- used by destroy
 * for any requests left in the queue at shutdown. */
static void write_request_release(write_request_t *req)
{
  if (req == NULL || req->picture == NULL) return;
  free_storable_picture(req->picture);
  req->picture = NULL;
}

void jm_write_queue_destroy(jm_write_queue_t *q)
{
  if (q == NULL) return;

  while (q->count > 0)
  {
    write_request_release(&q->buf[q->head]);
    q->head = (q->head + 1) % q->cap;
    q->count--;
  }

  jm_cond_destroy (&q->idle);
  jm_cond_destroy (&q->not_empty);
  jm_cond_destroy (&q->not_full);
  jm_mutex_destroy(&q->m);
  free(q->buf);
  free(q);
}

int jm_write_queue_push(jm_write_queue_t *q, const write_request_t *req)
{
  if (q == NULL || req == NULL) return -1;

  jm_mutex_lock(&q->m);
  while (q->count == q->cap && !q->closed)
    jm_cond_wait(&q->not_full, &q->m);
  if (q->closed)
  {
    jm_mutex_unlock(&q->m);
    return -1;
  }
  q->buf[q->tail] = *req;
  q->tail = (q->tail + 1) % q->cap;
  q->count++;
  jm_cond_signal(&q->not_empty);
  jm_mutex_unlock(&q->m);
  return 0;
}

int jm_write_queue_pop(jm_write_queue_t *q, write_request_t *out)
{
  if (q == NULL || out == NULL) return -1;

  jm_mutex_lock(&q->m);
  while (q->count == 0 && !q->closed)
    jm_cond_wait(&q->not_empty, &q->m);
  if (q->count == 0)
  {
    jm_mutex_unlock(&q->m);
    return -1;
  }
  *out = q->buf[q->head];
  memset(&q->buf[q->head], 0, sizeof(write_request_t));
  q->head = (q->head + 1) % q->cap;
  q->count--;
  q->inflight = 1;
  jm_cond_signal(&q->not_full);
  jm_mutex_unlock(&q->m);
  return 0;
}

/* Block until the queue is empty AND the writer is not currently
 * processing a request. After this returns, all writes that were
 * pushed before the call have completed -- pY/pU/pV in their
 * pDecPic slots is fully written and bValid has been cleared. */
void jm_write_queue_drain(jm_write_queue_t *q)
{
  if (q == NULL) return;
  jm_mutex_lock(&q->m);
  while (q->count > 0 || q->inflight)
    jm_cond_wait(&q->idle, &q->m);
  jm_mutex_unlock(&q->m);
}

void jm_write_queue_close(jm_write_queue_t *q)
{
  if (q == NULL) return;

  jm_mutex_lock(&q->m);
  q->closed = 1;
  jm_cond_broadcast(&q->not_empty);
  jm_cond_broadcast(&q->not_full);
  jm_mutex_unlock(&q->m);
}

/* ===================================================================== */
/*                              WRITER THREAD                              */
/* ===================================================================== */

/* Drain helper lives in output.c (has access to static write_out_picture
 * and the io_write helper). */
extern void jm_writer_drain_request(write_request_t *req);

static void *writer_thread_main(void *arg)
{
  VideoParameters *p_Vid = (VideoParameters *)arg;
  jm_write_queue_t *q = p_Vid->write_queue;
  write_request_t req;

  while (jm_write_queue_pop(q, &req) == 0)
  {
    jm_writer_drain_request(&req);
    /* drain_request runs write_out_picture (which itself calls
     * free_storable_picture, dropping the writer's reference).
     * Mark this request done and wake any thread waiting in
     * jm_write_queue_drain. */
    jm_mutex_lock(&q->m);
    q->inflight = 0;
    if (q->count == 0)
      jm_cond_broadcast(&q->idle);
    jm_mutex_unlock(&q->m);
  }
  return NULL;
}

void jm_writer_start(VideoParameters *p_Vid)
{
  if (p_Vid == NULL || p_Vid->write_queue == NULL) return;
  p_Vid->writer_thread_running = 1;
  if (jm_thread_create(&p_Vid->writer_thread, writer_thread_main, p_Vid) != 0)
  {
    p_Vid->writer_thread_running = 0;
    no_mem_exit("jm_writer_start: failed to create thread");
  }
}

void jm_writer_stop(VideoParameters *p_Vid)
{
  if (p_Vid == NULL || !p_Vid->writer_thread_running) return;
  if (p_Vid->write_queue)
    jm_write_queue_close(p_Vid->write_queue);
  jm_thread_join(p_Vid->writer_thread);
  p_Vid->writer_thread_running = 0;
}

void jm_writer_drain(VideoParameters *p_Vid)
{
  if (p_Vid == NULL || p_Vid->write_queue == NULL) return;
  jm_write_queue_drain(p_Vid->write_queue);
}
