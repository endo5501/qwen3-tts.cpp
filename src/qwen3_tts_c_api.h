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

// Lifecycle
QWEN3_TTS_API qwen3_tts_ctx * qwen3_tts_init(const char * model_dir, int n_threads);
QWEN3_TTS_API int              qwen3_tts_is_loaded(const qwen3_tts_ctx * ctx);
QWEN3_TTS_API void             qwen3_tts_free(qwen3_tts_ctx * ctx);

// Configuration
QWEN3_TTS_API void qwen3_tts_set_language(qwen3_tts_ctx * ctx, int language_id);

// Synthesis (results stored internally; 0 = success, -1 = error)
QWEN3_TTS_API int qwen3_tts_synthesize(qwen3_tts_ctx * ctx, const char * text);
QWEN3_TTS_API int qwen3_tts_synthesize_with_voice(qwen3_tts_ctx * ctx,
                                                    const char * text,
                                                    const char * ref_wav_path);
QWEN3_TTS_API int qwen3_tts_synthesize_with_embedding(qwen3_tts_ctx * ctx,
                                                        const char * text,
                                                        const float * emb_data,
                                                        int emb_size);

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
