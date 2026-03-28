#include "qwen3_tts_c_api.h"
#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
static std::wstring utf8_to_wstring_api(const std::string & utf8) {
    if (utf8.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring wstr(len - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wstr[0], len);
    return wstr;
}
static FILE * fopen_utf8(const char * path, const char * mode) {
    std::wstring wpath = utf8_to_wstring_api(path);
    std::wstring wmode;
    for (const char * p = mode; *p; ++p) wmode += (wchar_t)*p;
    return _wfopen(wpath.c_str(), wmode.c_str());
}
#else
static FILE * fopen_utf8(const char * path, const char * mode) {
    return fopen(path, mode);
}
#endif

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
    params.print_timing   = true;
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
    params.print_timing   = true;
    params.language_id    = ctx->language_id;

    ctx->last_result = ctx->tts.synthesize_with_voice(text, ref_wav_path, params);
    if (!ctx->last_result.success) {
        ctx->last_error = ctx->last_result.error_msg;
        return -1;
    }
    ctx->last_error.clear();
    return 0;
}

int qwen3_tts_synthesize_with_embedding(qwen3_tts_ctx * ctx,
                                         const char * text,
                                         const float * emb_data,
                                         int emb_size) {
    if (!ctx || !text) return -1;

    qwen3_tts::tts_params params;
    params.print_progress = false;
    params.print_timing   = true;
    params.language_id    = ctx->language_id;

    ctx->last_result = ctx->tts.synthesize_with_embedding(text, emb_data, emb_size, params);
    if (!ctx->last_result.success) {
        ctx->last_error = ctx->last_result.error_msg;
        return -1;
    }
    ctx->last_error.clear();
    return 0;
}

int qwen3_tts_extract_speaker_embedding(qwen3_tts_ctx * ctx,
                                          const char * ref_wav_path,
                                          float ** out_data,
                                          int * out_size) {
    if (!ctx || !ref_wav_path || !out_data || !out_size) return -1;

    *out_data = nullptr;
    *out_size = 0;

    std::vector<float> embedding;
    if (!ctx->tts.extract_speaker_embedding(ref_wav_path, embedding)) {
        ctx->last_error = ctx->tts.get_error();
        return -1;
    }

    *out_size = (int)embedding.size();
    *out_data = (float *)malloc(embedding.size() * sizeof(float));
    if (!*out_data) {
        ctx->last_error = "Failed to allocate memory for embedding";
        return -1;
    }
    memcpy(*out_data, embedding.data(), embedding.size() * sizeof(float));

    ctx->last_error.clear();
    return 0;
}

int qwen3_tts_save_speaker_embedding(const char * path,
                                       const float * data,
                                       int size) {
    if (!path || !data || size <= 0) return -1;

    FILE * fp = fopen_utf8(path, "wb");
    if (!fp) return -1;

    size_t written = fwrite(data, sizeof(float), (size_t)size, fp);
    fclose(fp);

    return (written == (size_t)size) ? 0 : -1;
}

int qwen3_tts_load_speaker_embedding(const char * path,
                                       float ** out_data,
                                       int * out_size) {
    if (!path || !out_data || !out_size) return -1;

    *out_data = nullptr;
    *out_size = 0;

    FILE * fp = fopen_utf8(path, "rb");
    if (!fp) return -1;

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= 0 || file_size % sizeof(float) != 0) {
        fclose(fp);
        return -1;
    }

    int n_floats = (int)(file_size / sizeof(float));
    float * data = (float *)malloc(file_size);
    if (!data) {
        fclose(fp);
        return -1;
    }

    size_t read = fread(data, sizeof(float), (size_t)n_floats, fp);
    fclose(fp);

    if (read != (size_t)n_floats) {
        free(data);
        return -1;
    }

    *out_data = data;
    *out_size = n_floats;
    return 0;
}

void qwen3_tts_free_speaker_embedding(float * data) {
    free(data);
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
