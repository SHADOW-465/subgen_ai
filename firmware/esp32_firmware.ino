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
 *     {"ok":true,"frames":<int>,"rms":<float>,"mfcc_mean":[f0,…,f11],"mfcc_var":[f0,…,f11]}
 *   Error:
 *     {"ok":false,"error":"<reason>"}
 *
 * ── ALGORITHM (mirrors subgen_ai/core/esp32_validator.py) ─────────────────
 *   For each 25 ms frame (400 samples) stepped by 10 ms (160 samples):
 *     1. Apply Hanning window (matches Python np.hanning)
 *     2. Zero-pad to 512 samples
 *     3. 512-point real FFT  → power spectrum (|X|² / N_FFT)
 *     4. Multiply by 26-band triangular mel filterbank (0–8 kHz)
 *     5. log10(energy + 1e-9)
 *     6. DCT-II with ortho normalisation → take first 12 coefficients
 *   Aggregate mean + variance across all frames.
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
#define BAUD_RATE     460800

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

// ── Compute MFCC for one audio frame ─────────────────────────────────────
static float frame_buf[N_FFT];
static float mel_e[N_MELS];

static void compute_frame_mfcc(const float *frame, float *mfcc_out) {
    // Hanning window + zero-pad
    memset(frame_buf, 0, sizeof(frame_buf));
    for (int i = 0; i < WIN_LENGTH; i++)
        frame_buf[i] = frame[i] * hanning[i];

    // FFT → power spectrum
    fft_real(frame_buf, N_FFT);
    int n_bins = N_FFT / 2 + 1;
    float power[n_bins];
    for (int k = 0; k < n_bins; k++) {
        float re = fft_buf[2*k], im = fft_buf[2*k+1];
        power[k] = (re*re + im*im) / N_FFT;
    }

    // Mel filterbank energies + log compression
    for (int m = 0; m < N_MELS; m++) {
        float e = 0.0f;
        for (int k = 0; k < n_bins; k++)
            e += mel_fb[m][k] * power[k];
        mel_e[m] = log10f(e + 1e-9f);
    }

    // DCT-II → 12 coefficients
    for (int c = 0; c < N_MFCC; c++)
        mfcc_out[c] = dct_ortho(mel_e, N_MELS, c);
}

// ── Static audio + accumulator buffers ───────────────────────────────────
static int16_t pcm_buf[MAX_SAMPLES];
static float   mfcc_sum[N_MFCC];
static float   mfcc_sq[N_MFCC];
static float   frame_mfcc[N_MFCC];

// ── Main pipeline ─────────────────────────────────────────────────────────
static void process_audio(int n_samples) {
    if (!fb_ready) build_mel_filterbank();

    // Normalise to float32 [-1, 1]
    // Compute RMS
    double rms_acc = 0.0;
    for (int i = 0; i < n_samples; i++)
        rms_acc += (double)pcm_buf[i] * pcm_buf[i];
    float rms = sqrtf((float)(rms_acc / n_samples)) / 32768.0f;

    memset(mfcc_sum, 0, sizeof(mfcc_sum));
    memset(mfcc_sq,  0, sizeof(mfcc_sq));
    int n_frames = 0;

    for (int start = 0; start + WIN_LENGTH <= n_samples; start += HOP_LENGTH) {
        // Convert int16 window to float32
        float win_f[WIN_LENGTH];
        for (int i = 0; i < WIN_LENGTH; i++)
            win_f[i] = pcm_buf[start + i] / 32768.0f;

        compute_frame_mfcc(win_f, frame_mfcc);

        for (int c = 0; c < N_MFCC; c++) {
            mfcc_sum[c] += frame_mfcc[c];
            mfcc_sq[c]  += frame_mfcc[c] * frame_mfcc[c];
        }
        n_frames++;
    }

    // Build JSON response
    StaticJsonDocument<1024> doc;
    if (n_frames == 0) {
        doc["ok"]    = false;
        doc["error"] = "no frames";
    } else {
        doc["ok"]     = true;
        doc["frames"] = n_frames;
        doc["rms"]    = rms;
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
