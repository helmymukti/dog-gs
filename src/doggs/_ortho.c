/* DOG-GS reference C backend: front-to-back compositing of orthographic Gaussian splats.
 * Mirrors doggs.kernel._composite exactly. Build: doggs build-c  (or gcc -O3 -fopenmp -shared -fPIC). */
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define T_MIN 1e-4f
#define A_MIN (1.0f / 255.0f)
#define A_MAX 0.99f
#define Q_MAX 18.0f

#ifdef _WIN32
__declspec(dllexport)
#endif
void composite(int n, const double *px, const double *py, const float *z,
               const float *a, const float *b, const float *c, const float *rad,
               const float *op, const float *rgb,
               double x0, double y0, int H, int W,
               float *out_rgb, float *out_T, float *out_zw, float *out_w, float *out_zmed)
{
#pragma omp parallel
    {
        int nt = 1, id = 0;
#ifdef _OPENMP
        nt = omp_get_num_threads(); id = omp_get_thread_num();
#endif
        int r0 = (int)((long long)H * id / nt), r1 = (int)((long long)H * (id + 1) / nt);
        for (int i = 0; i < n; i++) {
            double cx = px[i] - x0 - 0.5, cy = py[i] - y0 - 0.5;
            float R = rad[i];
            int ya = (int)floor(cy - R), yb = (int)ceil(cy + R);
            if (yb < r0 || ya >= r1) continue;
            int xa = (int)floor(cx - R), xb = (int)ceil(cx + R);
            if (xb < 0 || xa >= W) continue;
            if (ya < r0) ya = r0;
            if (yb > r1 - 1) yb = r1 - 1;
            if (xa < 0) xa = 0;
            if (xb > W - 1) xb = W - 1;
            float A = a[i], B = b[i], C = c[i], o = op[i], zi = z[i];
            float cr = rgb[3 * i], cg = rgb[3 * i + 1], cb = rgb[3 * i + 2];
            for (int y = ya; y <= yb; y++) {
                float dy = (float)(y - cy);
                long long base = (long long)y * W;
                for (int x = xa; x <= xb; x++) {
                    long long p = base + x;
                    float T = out_T[p];
                    if (T < T_MIN) continue;
                    float dx = (float)(x - cx);
                    float q = A * dx * dx + 2.f * B * dx * dy + C * dy * dy;
                    if (q > Q_MAX) continue;
                    float al = o * expf(-0.5f * q);
                    if (al < A_MIN) continue;
                    if (al > A_MAX) al = A_MAX;
                    float w = al * T;
                    out_rgb[3 * p] += w * cr;
                    out_rgb[3 * p + 1] += w * cg;
                    out_rgb[3 * p + 2] += w * cb;
                    out_zw[p] += w * zi;
                    out_w[p] += w;
                    float Tn = T * (1.f - al);
                    if (T >= 0.5f && Tn < 0.5f) out_zmed[p] = zi;
                    out_T[p] = Tn;
                }
            }
        }
    }
}
