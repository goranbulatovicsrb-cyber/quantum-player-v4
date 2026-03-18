#pragma once
#include <cmath>

// Second-order IIR (biquad) filter
// Implements peak EQ, low shelf, and high shelf filters
struct BiquadFilter {
    enum Type { PeakEQ, LowShelf, HighShelf, LowPass, HighPass };

    double b0=1, b1=0, b2=0, a1=0, a2=0;
    double x1=0, x2=0, y1=0, y2=0;

    void setParams(Type type, double freq, double sampleRate,
                   double gainDB, double Q = 1.41421356) {
        double A  = std::pow(10.0, gainDB / 40.0);
        double w0 = 2.0 * M_PI * freq / sampleRate;
        double cosW = std::cos(w0);
        double sinW = std::sin(w0);
        double alpha = sinW / (2.0 * Q);
        double a0;

        switch (type) {
        case PeakEQ: {
            b0 = 1 + alpha * A;
            b1 = -2 * cosW;
            b2 = 1 - alpha * A;
            a0 = 1 + alpha / A;
            a1 = -2 * cosW;
            a2 = 1 - alpha / A;
            break;
        }
        case LowShelf: {
            double sqA = std::sqrt(A);
            double aq  = 2.0 * sqA * alpha;
            b0 = A * ((A+1) - (A-1)*cosW + aq);
            b1 = 2 * A * ((A-1) - (A+1)*cosW);
            b2 = A * ((A+1) - (A-1)*cosW - aq);
            a0 = (A+1) + (A-1)*cosW + aq;
            a1 = -2 * ((A-1) + (A+1)*cosW);
            a2 = (A+1) + (A-1)*cosW - aq;
            break;
        }
        case HighShelf: {
            double sqA = std::sqrt(A);
            double aq  = 2.0 * sqA * alpha;
            b0 = A * ((A+1) + (A-1)*cosW + aq);
            b1 = -2 * A * ((A-1) + (A+1)*cosW);
            b2 = A * ((A+1) + (A-1)*cosW - aq);
            a0 = (A+1) - (A-1)*cosW + aq;
            a1 = 2 * ((A-1) - (A+1)*cosW);
            a2 = (A+1) - (A-1)*cosW - aq;
            break;
        }
        default: break;
        }
        b0/=a0; b1/=a0; b2/=a0; a1/=a0; a2/=a0;
    }

    inline double process(double x) {
        double y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        x2=x1; x1=x; y2=y1; y1=y;
        return y;
    }

    void reset() { x1=x2=y1=y2=0; }
};
