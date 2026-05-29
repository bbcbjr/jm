# library
set( LIB_NAME lcommon )

# NATVIS files for Visual Studio
if( MSVC )
  file( GLOB NATVIS_FILES "../../VisualStudio/*.natvis" )
endif()

# get source files
file( GLOB BASE_SRC_FILES "*.c" )

# get include files
file( GLOB BASE_INC_FILES "*.h" )


# get all source files
set( SRC_FILES ${BASE_SRC_FILES} )

# get all include files
set( INC_FILES ${BASE_INC_FILES} )

# Per-TU compile flags for SIMD ISA tiers.
# transform_sse.c is SSE2-only -- SSE2 is the x86-64 baseline on every
# compiler, so no extra flag is required.
# transform_avx2.c uses _mm256_* / immintrin.h. MSVC tolerates these in
# any TU but GCC and Clang error out without -mavx2. Setting the flag
# per-TU lets the rest of lcommon stay buildable without AVX2 being
# enabled globally (which would risk the compiler auto-vectorizing
# unrelated code with AVX2 ops and failing on pre-Haswell CPUs).
if( MSVC )
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/transform_avx2.c
    PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
  )
else()
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/transform_avx2.c
    PROPERTIES COMPILE_OPTIONS "-mavx2"
  )
endif()

# library
add_library( ${LIB_NAME} STATIC ${SRC_FILES} ${INC_FILES} ${NATVIS_FILES} )

if( SET_ENABLE_TRACING )
  if( ENABLE_TRACING )
    target_compile_definitions( ${LIB_NAME} PUBLIC ENABLE_TRACING=1 )
  else()
    target_compile_definitions( ${LIB_NAME} PUBLIC ENABLE_TRACING=0 )
  endif()
endif()

target_include_directories( ${LIB_NAME} PUBLIC . .. )
target_link_libraries( ${LIB_NAME} Threads::Threads )

# example: place header files in different folders
source_group( "Natvis Files" FILES ${NATVIS_FILES} )

# set the folder where to place the projects
set_target_properties( ${LIB_NAME} PROPERTIES FOLDER lib )
