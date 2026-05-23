#ifndef _JM_DEFINES_H_
#define _JM_DEFINES_H_

#if defined(_MSC_VER)
#define JM_FORCEINLINE static __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define JM_FORCEINLINE static inline __attribute__((always_inline))
#else
#define JM_FORCEINLINE static inline
#endif

#if defined(_MSC_VER)
// No direct hint in MSVC pre-VS2022; rely on PGO or [[likely]] in C++20
#define JM_LIKELY(x) (x)
#define JM_UNLIKELY(x) (x)
#else
#define JM_LIKELY(x) __builtin_expect(!!(x), 1)
#define JM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#endif

