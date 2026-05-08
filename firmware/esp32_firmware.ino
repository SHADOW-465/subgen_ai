/**
 * SubGEN AI — ESP32 MFCC Fingerprint Firmware
 * =============================================
 * Board  : ESP32 Dev Module
 * Baud   : 460800
 * Library: ArduinoJson v6+  (install via Arduino Library Manager)
 *
 * ── SERIAL PROTOCOL ──────────────────────────────────────────────────────
 *
 * HOST → ESP32  (binary frame):
 *   Byte 0   : 0xAA  (header byte 1)
 *   Byte 1   : 0x55  (header byte 2)
 *   Byte 2-3 : N     (uint16 big-endian — number of int16 PCM samples)
 *   Byte 4…  : N × 2 bytes of PCM int16 little-endian samples
 *              (mono, 16 kHz, range -32768 … 32767)
 *              Max N = 32000 (2 seconds at 16 kHz)
 *
 * ESP32 → HOST  (single JSON line, UTF-8, terminated with '\n'):
 *   Success:
 *     {"ok":true,"frames":<int>,"rms":<float>,"snr_db":<float>,
 *      "mfcc_mean":[f0,…,f11],"mfcc_var":[f0,…,f11]}
 *   Error:
 *     {"ok":false,"error":"<reason>"}
 *
 * ── ALGORITHM (mirrors subgen_ai/core/esp32_validator.py) ─────────────────
 *   Two-pass spectral subtraction:
 *
 *   Pass 1 — noise floor estimation:
 *     For each 25 ms frame stepped by 10 ms:
 *       1. Apply Hanning window + zero-pad to 512 samples
 *       2. 512-point real FFT → power spectrum (|X|² / N_FFT)
 *       3. Multiply by 26-band triangular mel filterbank → raw mel energies
 *       4. Classify frame: RMS < RMS_SILENCE → noise, else speech
 *     noise_floor[m] = mean raw mel energy over all noise frames, per band.
 *     snr_db = 10*log10(mean_speech_mel / mean_noise_mel), clipped [-20, 60].
 *
 *   Pass 2 — denoised MFCC:
 *     For each frame:
 *       1. Recompute raw mel energies (same as Pass 1)
 *       2. mel_denoised[m] = max(mel[m] - noise_floor[m], 1e-9)
 *       3. log10(mel_denoised[m])          ← log AFTER subtraction
 *       4. DCT-II with ortho normalisation → first 12 coefficients
 *     Aggregate mean + variance across all frames.
 *
 * ── NOTES ────────────────────────────────────────────────────────────────
 *  • This file is reference documentation. The Python app does NOT
 *    compile or flash it — do that manually via the Arduino IDE.
 *  • Increase Serial.setRxBufferSize before Serial.begin for large frames.
 *  • The mel filterbank is pre-computed at startup to save runtime cycles.
 */

#include <Arduino.h>
#include <ArduinoJson.h>
#include <math.h>

// ── Constants (must match Python side) ────────────────────────────────────
#define SAMPLE_RATE   16000
#define N_FFT         512
#define HOP_LENGTH    160       // 10 ms
#define WIN_LENGTH    400       // 25 ms
#define N_MFCC        12
#define N_MELS        26
#define FMIN_HZ       0.0f
#define FMAX_HZ       8000.0f
#define MAX_SAMPLES   32000     // 2 s at 16 kHz
#define BAUD_RATE     2000000   // 2 Mbps — requires CH340/CH341 bridge chip
                               // (CP2102 max is ~921600; use CH340 boards)
                               // At 2 Mbps: 64 KB audio transfers in ~0.32 s
                               // vs ~1.39 s at 460800 — critical for real-time use
#define RMS_SILENCE   0.002f    // frames below this normalised RMS = noise frames

// ── Mel filterbank (pre-computed once) ────────────────────────────────────
static float mel_fb[N_MELS][N_FFT / 2 + 1];
static bool  fb_ready = false;

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void build_mel_filterbank() {
    int   n_bins = N_FFT / 2 + 1;
    float mel_min = hz_to_mel(FMIN_HZ);
    float mel_max = hz_to_mel(FMAX_HZ);

    float mel_pts[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++)
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (N_MELS + 1);

    int bin_pts[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++)
        bin_pts[i] = (int)floorf((N_FFT + 1) * mel_to_hz(mel_pts[i]) / SAMPLE_RATE);

    memset(mel_fb, 0, sizeof(mel_fb));
    for (int m = 1; m <= N_MELS; m++) {
        int fl = bin_pts[m - 1], fc = bin_pts[m], fr = bin_pts[m + 1];
        for (int k = fl; k < fc; k++)
            if (fc != fl && k < n_bins)
                mel_fb[m - 1][k] = (float)(k - fl) / (fc - fl);
        for (int k = fc; k < fr; k++)
            if (fr != fc && k < n_bins)
                mel_fb[m - 1][k] = (float)(fr - k) / (fr - fc);
    }
    fb_ready = true;
}

// ── Hanning window ─────────────────────────────────────────────────────────
static float hanning[WIN_LENGTH];

static void build_hanning() {
    for (int i = 0; i < WIN_LENGTH; i++)
        hanning[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (WIN_LENGTH - 1)));
}

// ── Minimal real FFT (Cooley-Tukey radix-2, in-place) ────────────────────
// Re-uses a scratch buffer; operates on float pairs [re, im] interleaved.
static float fft_buf[N_FFT * 2];  // [re0, im0, re1, im1, ...]

static void fft_real(float *re_in, int n) {
    // Copy into complex buffer (imaginary = 0)
    for (int i = 0; i < n; i++) {
        fft_buf[2 * i]     = re_in[i];
        fft_buf[2 * i + 1] = 0.0f;
    }
    // Bit-reversal permutation
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = fft_buf[2*i]; fft_buf[2*i] = fft_buf[2*j]; fft_buf[2*j] = tr;
            float ti = fft_buf[2*i+1]; fft_buf[2*i+1] = fft_buf[2*j+1]; fft_buf[2*j+1] = ti;
        }
    }
    // Butterfly stages
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int u = 2*(i+k), v = 2*(i+k+len/2);
                float tr = cur_re*fft_buf[v]   - cur_im*fft_buf[v+1];
                float ti = cur_re*fft_buf[v+1] + cur_im*fft_buf[v];
                fft_buf[v]   = fft_buf[u]   - tr;
                fft_buf[v+1] = fft_buf[u+1] - ti;
                fft_buf[u]   += tr;
                fft_buf[u+1] += ti;
                float new_re = cur_re*wre - cur_im*wim;
                cur_im       = cur_re*wim + cur_im*wre;
                cur_re       = new_re;
            }
        }
    }
}

// ── DCT-II (ortho) ────────────────────────────────────────────────────────
static float dct_ortho(float *x, int n, int k) {
    // Single coefficient k of DCT-II with ortho normalisation
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += x[i] * cosf(M_PI * k * (2*i + 1) / (2.0f * n));
    float scale = (k == 0) ? sqrtf(1.0f / n) : sqrtf(2.0f / n);
    return scale * sum;
}

// ── Raw mel energy extraction (no log, no DCT) ───────────────────────────
// Used by both passes. Log compression and DCT are applied in Pass 2 AFTER
// spectral subtraction, so they must not happen inside this function.
static float frame_buf[N_FFT];
static float mel_e[N_MELS];

static void compute_frame_mel_raw(const float *frame, float *mel_out) {
    // Hanning window + zero-pad to N_FFT
    memset(frame_buf, 0, sizeof(frame_buf));
    for (int i = 0; i < WIN_LENGTH; i++)
        frame_buf[i] = frame[i] * hanning[i];

    // 512-point real FFT
    fft_real(frame_buf, N_FFT);

    // Mel filterbank energies — raw (pre-log)
    int n_bins = N_FFT / 2 + 1;
    for (int m = 0; m < N_MELS; m++) {
        float e = 0.0f;
        for (int k = 0; k < n_bins; k++) {
            float re = fft_buf[2*k], im = fft_buf[2*k+1];
            e += mel_fb[m][k] * (re*re + im*im) / N_FFT;
        }
        mel_out[m] = e;  // raw energy — log applied after spectral subtraction
    }
}

// ── Static audio + accumulator buffers ───────────────────────────────────
static int16_t pcm_buf[MAX_SAMPLES];
static float   mfcc_sum[N_MFCC];
static float   mfcc_sq[N_MFCC];

// ── Main pipeline — two-pass spectral subtraction ────────────────────────
static void process_audio(int n_samples) {
    if (!fb_ready) build_mel_filterbank();

    // Overall clip RMS (for JSON metadata)
    double rms_acc = 0.0;
    for (int i = 0; i < n_samples; i++)
        rms_acc += (double)pcm_buf[i] * pcm_buf[i];
    float rms = sqrtf((float)(rms_acc / n_samples)) / 32768.0f;

    // ── Pass 1: per-band noise floor estimation ───────────────────────────
    float noise_floor[N_MELS];
    memset(noise_floor, 0, sizeof(noise_floor));
    int   n_noise = 0, n_speech = 0;
    float sum_s   = 0.0f, sum_n = 0.0f;

    for (int s = 0; s + WIN_LENGTH <= n_samples; s += HOP_LENGTH) {
        float win_f[WIN_LENGTH];
        float rms_f = 0.0f;
        for (int i = 0; i < WIN_LENGTH; i++) {
            win_f[i] = pcm_buf[s + i] / 32768.0f;
            rms_f += win_f[i] * win_f[i];
        }
        rms_f = sqrtf(rms_f / WIN_LENGTH);

        compute_frame_mel_raw(win_f, mel_e);

        // Per-frame mean mel energy for SNR estimation
        float fmean = 0.0f;
        for (int m = 0; m < N_MELS; m++) fmean += mel_e[m];
        fmean /= N_MELS;

        if (rms_f < RMS_SILENCE) {
            // Noise frame — accumulate toward noise floor
            for (int m = 0; m < N_MELS; m++) noise_floor[m] += mel_e[m];
            sum_n += fmean;
            n_noise++;
        } else {
            // Speech frame
            sum_s += fmean;
            n_speech++;
        }
    }

    // Normalise noise floor to mean (all-zero if no silence detected)
    if (n_noise > 0)
        for (int m = 0; m < N_MELS; m++) noise_floor[m] /= n_noise;

    // Mel-domain SNR
    float snr_db;
    if (n_speech > 0 && n_noise > 0) {
        float ms = sum_s / n_speech;
        float mn = (sum_n / n_noise < 1e-10f) ? 1e-10f : sum_n / n_noise;
        snr_db = 10.0f * log10f(ms / mn);
        if (snr_db < -20.0f) snr_db = -20.0f;
        if (snr_db >  60.0f) snr_db =  60.0f;
    } else if (n_speech > 0) {
        snr_db = 60.0f;   // clean clip — no noise frames
    } else {
        snr_db = -20.0f;  // all noise or empty
    }

    // ── Pass 2: subtract noise floor → log10 → DCT-II → accumulate ──────
    memset(mfcc_sum, 0, sizeof(mfcc_sum));
    memset(mfcc_sq,  0, sizeof(mfcc_sq));
    int n_frames = 0;

    for (int s = 0; s + WIN_LENGTH <= n_samples; s += HOP_LENGTH) {
        float win_f[WIN_LENGTH];
        for (int i = 0; i < WIN_LENGTH; i++)
            win_f[i] = pcm_buf[s + i] / 32768.0f;

        compute_frame_mel_raw(win_f, mel_e);

        // Spectral subtraction + log compression
        for (int m = 0; m < N_MELS; m++) {
            mel_e[m] -= noise_floor[m];
            if (mel_e[m] < 1e-9f) mel_e[m] = 1e-9f;  // floor before log
            mel_e[m] = log10f(mel_e[m]);
        }

        // DCT-II → accumulate mean and variance
        for (int c = 0; c < N_MFCC; c++) {
            float coeff = dct_ortho(mel_e, N_MELS, c);
            mfcc_sum[c] += coeff;
            mfcc_sq[c]  += coeff * coeff;
        }
        n_frames++;
    }

    // ── Build JSON response ───────────────────────────────────────────────
    StaticJsonDocument<1536> doc;
    if (n_frames == 0) {
        doc["ok"]    = false;
        doc["error"] = "no frames";
    } else {
        doc["ok"]     = true;
        doc["frames"] = n_frames;
        doc["rms"]    = rms;
        doc["snr_db"] = snr_db;
        JsonArray mean_arr = doc.createNestedArray("mfcc_mean");
        JsonArray var_arr  = doc.createNestedArray("mfcc_var");
        for (int c = 0; c < N_MFCC; c++) {
            float mean = mfcc_sum[c] / n_frames;
            float var  = (mfcc_sq[c] / n_frames) - (mean * mean);
            mean_arr.add(mean);
            var_arr.add(var);
        }
    }
    serializeJson(doc, Serial);
    Serial.println();   // terminate line
}

// ── setup / loop ──────────────────────────────────────────────────────────
void setup() {
    Serial.setRxBufferSize(MAX_SAMPLES * 2 + 64);
    Serial.begin(BAUD_RATE);
    build_hanning();
    build_mel_filterbank();
}

void loop() {
    // Wait for header: 0xAA 0x55
    if (Serial.available() < 2) return;
    uint8_t b0 = Serial.read();
    if (b0 != 0xAA) return;
    uint8_t b1 = Serial.read();
    if (b1 != 0x55) return;

    // Read 2-byte big-endian sample count
    while (Serial.available() < 2) delay(1);
    uint8_t nh = Serial.read(), nl = Serial.read();
    uint16_t n = ((uint16_t)nh << 8) | nl;

    if (n > MAX_SAMPLES) {
        // Drain and report error
        for (uint32_t i = 0; i < (uint32_t)n * 2; i++) {
            while (!Serial.available()) delay(1);
            Serial.read();
        }
        StaticJsonDocument<128> err;
        err["ok"] = false;
        err["error"] = "frame too large";
        serializeJson(err, Serial);
        Serial.println();
        return;
    }

    // Read PCM bytes
    uint32_t bytes_to_read = (uint32_t)n * 2;
    uint32_t bytes_read    = 0;
    uint8_t *dst = (uint8_t *)pcm_buf;
    unsigned long t_start = millis();
    while (bytes_read < bytes_to_read) {
        if (Serial.available()) {
            dst[bytes_read++] = Serial.read();
        } else if (millis() - t_start > 3000) {
            // Timeout
            StaticJsonDocument<128> err;
            err["ok"] = false;
            err["error"] = "read timeout";
            serializeJson(err, Serial);
            Serial.println();
            return;
        }
    }

    process_audio(n);
}
