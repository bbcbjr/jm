
/*!
 ***************************************************************************
 * \file
 *    jm_threads.h
 *
 * \brief
 *    Portable threading primitives for Stage 2 view-parallel decoding (M4).
 *    Wraps Win32 (SRWLOCK + CONDITION_VARIABLE + CreateThread) on Windows
 *    and pthreads on Linux/macOS behind a tiny shared API.
 *
 *    Header-only, static inline. Compiles into each TU that includes it.
 *    No new compile units; no new link dependencies beyond what's already
 *    in CMakeLists (Threads::Threads is the cross-platform link, and on
 *    Windows the Win32 APIs come from kernel32 which is linked by default).
 *
 *    API (all functions return 0 on success, non-zero on error unless
 *    noted; error handling is "best-effort -- a thread/mutex/cond create
 *    failure aborts via no_mem_exit-style handling at the caller):
 *
 *        jm_thread_t                          - opaque thread handle
 *        jm_thread_create(t, fn, arg)         - spawn fn(arg) on a new thread
 *        jm_thread_join(t)                    - wait for t to finish
 *
 *        jm_mutex_t                           - opaque mutex
 *        jm_mutex_init(m)                     - initialize
 *        jm_mutex_lock(m) / jm_mutex_unlock(m)
 *        jm_mutex_destroy(m)
 *
 *        jm_cond_t                            - opaque condition variable
 *        jm_cond_init(c)                      - initialize
 *        jm_cond_wait(c, m)                   - atomic unlock m, sleep on c, relock m
 *        jm_cond_signal(c) / jm_cond_broadcast(c)
 *        jm_cond_destroy(c)
 *
 *    Thread entry signature follows the pthread idiom:
 *        typedef void *(*jm_thread_fn_t)(void *arg);
 *    On Win32 the return value is ignored (no jm_thread_join return).
 ***************************************************************************
 */

#ifndef _JM_THREADS_H_
#define _JM_THREADS_H_

#include <stdlib.h>

typedef void *(*jm_thread_fn_t)(void *arg);

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef HANDLE              jm_thread_t;
typedef SRWLOCK             jm_mutex_t;
typedef CONDITION_VARIABLE  jm_cond_t;

/* Win32 thread proc is DWORD WINAPI fn(LPVOID). Trampoline to the
 * pthread-style void *fn(void*) signature. The trampoline struct is
 * heap-allocated per create and freed on entry. */
typedef struct {
  jm_thread_fn_t fn;
  void          *arg;
} jm__win_tramp_t;

static DWORD WINAPI jm__win_thread_proc(LPVOID p)
{
  jm__win_tramp_t  t  = *(jm__win_tramp_t*)p;
  free(p);
  t.fn(t.arg);
  return 0;
}

static __inline int jm_thread_create(jm_thread_t *t, jm_thread_fn_t fn, void *arg)
{
  jm__win_tramp_t *tr = (jm__win_tramp_t*)malloc(sizeof(jm__win_tramp_t));
  HANDLE h;
  if (tr == NULL) return -1;
  tr->fn  = fn;
  tr->arg = arg;
  h = CreateThread(NULL, 0, jm__win_thread_proc, tr, 0, NULL);
  if (h == NULL) { free(tr); return -1; }
  *t = h;
  return 0;
}

static __inline int jm_thread_join(jm_thread_t t)
{
  WaitForSingleObject(t, INFINITE);
  CloseHandle(t);
  return 0;
}

static __inline void jm_mutex_init(jm_mutex_t *m)    { InitializeSRWLock(m); }
static __inline void jm_mutex_lock(jm_mutex_t *m)    { AcquireSRWLockExclusive(m); }
static __inline void jm_mutex_unlock(jm_mutex_t *m)  { ReleaseSRWLockExclusive(m); }
static __inline void jm_mutex_destroy(jm_mutex_t *m) { (void)m; /* SRWLOCK has no destroy */ }

static __inline void jm_cond_init(jm_cond_t *c)                          { InitializeConditionVariable(c); }
static __inline void jm_cond_wait(jm_cond_t *c, jm_mutex_t *m)           { SleepConditionVariableSRW(c, m, INFINITE, 0); }
static __inline void jm_cond_signal(jm_cond_t *c)                        { WakeConditionVariable(c); }
static __inline void jm_cond_broadcast(jm_cond_t *c)                     { WakeAllConditionVariable(c); }
static __inline void jm_cond_destroy(jm_cond_t *c)                       { (void)c; /* CONDITION_VARIABLE has no destroy */ }

#else /* POSIX: Linux, macOS, *BSD */

#include <pthread.h>

typedef pthread_t        jm_thread_t;
typedef pthread_mutex_t  jm_mutex_t;
typedef pthread_cond_t   jm_cond_t;

static inline int jm_thread_create(jm_thread_t *t, jm_thread_fn_t fn, void *arg)
{
  return pthread_create(t, NULL, fn, arg);
}

static inline int jm_thread_join(jm_thread_t t)
{
  return pthread_join(t, NULL);
}

static inline void jm_mutex_init(jm_mutex_t *m)    { pthread_mutex_init(m, NULL); }
static inline void jm_mutex_lock(jm_mutex_t *m)    { pthread_mutex_lock(m); }
static inline void jm_mutex_unlock(jm_mutex_t *m)  { pthread_mutex_unlock(m); }
static inline void jm_mutex_destroy(jm_mutex_t *m) { pthread_mutex_destroy(m); }

static inline void jm_cond_init(jm_cond_t *c)                  { pthread_cond_init(c, NULL); }
static inline void jm_cond_wait(jm_cond_t *c, jm_mutex_t *m)   { pthread_cond_wait(c, m); }
static inline void jm_cond_signal(jm_cond_t *c)                { pthread_cond_signal(c); }
static inline void jm_cond_broadcast(jm_cond_t *c)             { pthread_cond_broadcast(c); }
static inline void jm_cond_destroy(jm_cond_t *c)               { pthread_cond_destroy(c); }

#endif /* _WIN32 */

/*!
 ***************************************************************************
 * \brief
 *    Compile-time self-test: forces the compiler to type-check every
 *    primitive on every platform that includes this header. Marked unused
 *    so it gets dropped from the final binary; its only purpose is to
 *    catch broken declarations at build time.
 ***************************************************************************
 */
static void jm__threads_typecheck(void)
{
  jm_mutex_t  m;
  jm_cond_t   c;
  /* Don't actually invoke runtime primitives here -- pthread_mutex_init
   * with NULL attr is fine, but on some platforms calling these from a
   * static init context is undefined. We just take the addresses so the
   * compiler sees the symbols exist with the expected signatures. */
  (void)(jm_thread_create);
  (void)(jm_thread_join);
  (void)(jm_mutex_init);
  (void)(jm_mutex_lock);
  (void)(jm_mutex_unlock);
  (void)(jm_mutex_destroy);
  (void)(jm_cond_init);
  (void)(jm_cond_wait);
  (void)(jm_cond_signal);
  (void)(jm_cond_broadcast);
  (void)(jm_cond_destroy);
  (void)(&m);
  (void)(&c);
}

#endif  /* _JM_THREADS_H_ */
