#include "qwen3_tts_c_api.h"
#include "qwen3_tts.h"

#include <string>

struct qwen3_tts_ctx {
    qwen3_tts::Qwen3TTS   tts;
    qwen3_tts::tts_result  last_result;
    std::string            last_error;
    int32_t                language_id = 2058; // Japanese
};

qwen3_tts_ctx * qwen3_tts_init(const char * model_dir, int n_threads) {
    auto * ctx = new (std::nothrow) qwen3_tts_ctx();
    if (!ctx) {
        return nullptr;
    }

    if (!ctx->tts.load_models(model_dir)) {
        ctx->last_error = ctx->tts.get_error();
        delete ctx;
        return nullptr;
    }

    (void)n_threads; // stored in tts_params at synthesis time
    return ctx;
}

int qwen3_tts_is_loaded(const qwen3_tts_ctx * ctx) {
    if (!ctx) return 0;
    return ctx->tts.is_loaded() ? 1 : 0;
}

void qwen3_tts_free(qwen3_tts_ctx * ctx) {
    delete ctx;
}

void qwen3_tts_set_language(qwen3_tts_ctx * ctx, int language_id) {
    if (!ctx) return;
    ctx->language_id = language_id;
}

int qwen3_tts_synthesize(qwen3_tts_ctx * ctx, const char * text) {
    if (!ctx || !text) return -1;

    qwen3_tts::tts_params params;
    params.print_progress = false;
    params.print_timing   = false;
    params.language_id    = ctx->language_id;

    ctx->last_result = ctx->tts.synthesize(text, params);
    if (!ctx->last_result.success) {
        ctx->last_error = ctx->last_result.error_msg;
        return -1;
    }
    ctx->last_error.clear();
    return 0;
}

int qwen3_tts_synthesize_with_voice(qwen3_tts_ctx * ctx,
                                     const char * text,
                                     const char * ref_wav_path) {
    if (!ctx || !text || !ref_wav_path) return -1;

    qwen3_tts::tts_params params;
    params.print_progress = false;
    params.print_timing   = false;
    params.language_id    = ctx->language_id;

    ctx->last_result = ctx->tts.synthesize_with_voice(text, ref_wav_path, params);
    if (!ctx->last_result.success) {
        ctx->last_error = ctx->last_result.error_msg;
        return -1;
    }
    ctx->last_error.clear();
    return 0;
}

int qwen3_tts_synthesize_with_instruct(qwen3_tts_ctx * ctx,
                                        const char * text,
                                        const char * instruct) {
    if (!ctx || !text) return -1;

    qwen3_tts::tts_params params;
    params.print_progress = false;
    params.print_timing   = false;
    params.language_id    = ctx->language_id;
    if (instruct) params.instruct = instruct;

    ctx->last_result = ctx->tts.synthesize(text, params);
    if (!ctx->last_result.success) {
        ctx->last_error = ctx->last_result.error_msg;
        return -1;
    }
    ctx->last_error.clear();
    return 0;
}

int qwen3_tts_synthesize_with_voice_and_instruct(qwen3_tts_ctx * ctx,
                                                    const char * text,
                                                    const char * ref_wav_path,
                                                    const char * instruct) {
    if (!ctx || !text || !ref_wav_path) return -1;

    qwen3_tts::tts_params params;
    params.print_progress = false;
    params.print_timing   = false;
    params.language_id    = ctx->language_id;
    if (instruct) params.instruct = instruct;

    ctx->last_result = ctx->tts.synthesize_with_voice(text, ref_wav_path, params);
    if (!ctx->last_result.success) {
        ctx->last_error = ctx->last_result.error_msg;
        return -1;
    }
    ctx->last_error.clear();
    return 0;
}

const float * qwen3_tts_get_audio(const qwen3_tts_ctx * ctx) {
    if (!ctx || ctx->last_result.audio.empty()) return nullptr;
    return ctx->last_result.audio.data();
}

int qwen3_tts_get_audio_length(const qwen3_tts_ctx * ctx) {
    if (!ctx) return 0;
    return static_cast<int>(ctx->last_result.audio.size());
}

int qwen3_tts_get_sample_rate(const qwen3_tts_ctx * ctx) {
    if (!ctx) return 0;
    return ctx->last_result.sample_rate;
}

const char * qwen3_tts_get_error(const qwen3_tts_ctx * ctx) {
    if (!ctx) return "null context";
    return ctx->last_error.c_str();
}
