#ifndef QWEN3_TTS_C_API_H
#define QWEN3_TTS_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  define QWEN3_TTS_API __declspec(dllexport)
#else
#  define QWEN3_TTS_API __attribute__((visibility("default")))
#endif

typedef struct qwen3_tts_ctx qwen3_tts_ctx;

// Abort handle: holds the atomic abort flag with a lifetime that is independent
// of any synthesis context. It is created once (per TTS session) and outlives
// context init/reload/free, so aborting during a model reload never touches a
// freed context (fixes the F111 use-after-free).
typedef struct qwen3_tts_abort_handle qwen3_tts_abort_handle;

// Abort handle lifecycle
QWEN3_TTS_API qwen3_tts_abort_handle * qwen3_tts_create_abort_handle(void);
QWEN3_TTS_API void                     qwen3_tts_free_abort_handle(qwen3_tts_abort_handle * handle);

// Lifecycle.
// `abort_handle` may be null; when provided, its flag is checked during all
// synthesis on this context. The handle is owned by the caller and must outlive
// the context (it is NOT freed by qwen3_tts_free).
QWEN3_TTS_API qwen3_tts_ctx * qwen3_tts_init(const char * model_dir, int n_threads,
                                             qwen3_tts_abort_handle * abort_handle);
QWEN3_TTS_API int              qwen3_tts_is_loaded(const qwen3_tts_ctx * ctx);
QWEN3_TTS_API void             qwen3_tts_free(qwen3_tts_ctx * ctx);

// Configuration
QWEN3_TTS_API void qwen3_tts_set_language(qwen3_tts_ctx * ctx, int language_id);

// Abort (thread-safe, can be called from any thread). Operates on the abort
// handle, never on the context.
QWEN3_TTS_API void qwen3_tts_abort(qwen3_tts_abort_handle * handle);
QWEN3_TTS_API void qwen3_tts_reset_abort(qwen3_tts_abort_handle * handle);

// Synthesis (results stored internally; 0 = success, -1 = error)
// max_tokens: maximum audio frames to generate (0 or negative = use default 2048)
QWEN3_TTS_API int qwen3_tts_synthesize(qwen3_tts_ctx * ctx, const char * text,
                                         int max_tokens);
QWEN3_TTS_API int qwen3_tts_synthesize_with_voice(qwen3_tts_ctx * ctx,
                                                    const char * text,
                                                    const char * ref_wav_path,
                                                    int max_tokens);
QWEN3_TTS_API int qwen3_tts_synthesize_with_embedding(qwen3_tts_ctx * ctx,
                                                        const char * text,
                                                        const float * emb_data,
                                                        int emb_size,
                                                        int max_tokens);

// Speaker embedding extraction and file I/O (0 = success, -1 = error)
QWEN3_TTS_API int   qwen3_tts_extract_speaker_embedding(qwen3_tts_ctx * ctx,
                                                          const char * ref_wav_path,
                                                          float ** out_data,
                                                          int * out_size);
QWEN3_TTS_API int   qwen3_tts_save_speaker_embedding(const char * path,
                                                       const float * data,
                                                       int size);
QWEN3_TTS_API int   qwen3_tts_load_speaker_embedding(const char * path,
                                                       float ** out_data,
                                                       int * out_size);
QWEN3_TTS_API void  qwen3_tts_free_speaker_embedding(float * data);

// Result access
QWEN3_TTS_API const float * qwen3_tts_get_audio(const qwen3_tts_ctx * ctx);
QWEN3_TTS_API int            qwen3_tts_get_audio_length(const qwen3_tts_ctx * ctx);
QWEN3_TTS_API int            qwen3_tts_get_sample_rate(const qwen3_tts_ctx * ctx);
QWEN3_TTS_API const char *  qwen3_tts_get_error(const qwen3_tts_ctx * ctx);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_TTS_C_API_H
