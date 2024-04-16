/*
 * MIT License
 *
 * Copyright (c) 2023 [Your Name]
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ==============================================================================
 *
 * internal.hpp
 * Backend for fft.hpp
 *
 * ==============================================================================
 */

#pragma once
#include <complex>

namespace fft {
/*
 * ==============================================================================
 *
 * Internal API
 * We split the library in an internal part and an external part in order to
 * separate the implementation from the interface.
 *
 * ==============================================================================
 */
namespace internal {

// removed constexpr, unsupported in clang
template <typename T>
static T invSqrt2{T(1.0 / std::sqrt((T)2))};

template <typename C, bool F>
inline C rot90(C input)
{
    return F ? C(-input.imag(), input.real()) : C(input.imag(), -input.real());
}

template <typename T, typename C, bool F>
inline C rot45(C input)
{
    return F ? (C(input.real() - input.imag(), input.real() + input.imag())) *
                   invSqrt2<T>
             : (C(input.real() + input.imag(), -input.real() + input.imag())) *
                   invSqrt2<T>;
}

template <typename T, typename C, bool F>
inline C rot135(C input)
{
    return F ? (C(-input.real() - input.imag(), input.real() - input.imag())) *
                   invSqrt2<T>
             : (C(-input.real() + input.imag(), -input.real() - input.imag())) *
                   invSqrt2<T>;
}

template <typename T>
void populateCfftTwiddles(std::complex<T>* twiddleFactors, std::size_t nfft,
                          bool inverseTransform)
{
    T pi{std::acos((T)-1)};
    for (std::size_t i = 0; i < nfft; i++) {
        twiddleFactors[i] =
            inverseTransform
                ? std::exp(std::complex<T>(0, (T)i * (T)2 * pi / ((T)nfft)))
                : std::exp(std::complex<T>(0, -(T)i * (T)2 * pi / ((T)nfft)));
    }
}

template <typename T, bool F>
void transformRecursion(const std::complex<T>* in, std::complex<T>* out,
                        const std::complex<T>* twiddle, std::size_t offset,
                        std::size_t stride, std::size_t N, std::size_t mask)
{
    using C = std::complex<T>;
    switch (N) {
    case 1: {
        out[0] = in[offset & mask];
        break;
    }
    case 2: {
        C y0{in[(offset)&mask]};
        C y1{in[(offset + stride) & mask]};
        out[0] = y0 + y1;
        out[1] = y0 - y1;
        break;
    }
    case 4: {
        C y0{in[(offset)&mask]};
        C y1{in[(offset + stride) & mask]};
        C y2{in[(offset + 2 * stride) & mask]};
        C y3{in[(offset + 3 * stride) & mask]};
        // Calculate: data[i] += y1 + y2 + y3;
        out[0] = y0 + y1 + y2 + y3;
        // Calculate: data[i+N/4] = data[i] - j*y1 -y2 + j*y3;
        out[1] = y0 + rot90<C, F>(y1) - y2 - rot90<C, F>(y3);
        // Calculate: data[i+N/2] = data[i] - y1 + y2 - y3;
        out[2] = y0 - y1 + y2 - y3;
        // Calculate: data[i+3*N/4] = data[i] + j*y1 - y2 - j*y3;
        out[3] = y0 - rot90<C, F>(y1) - y2 + rot90<C, F>(y3);
        break;
    }
    case 8: {
        C y0{in[offset & mask]};
        C y1{in[(offset + stride) & mask]};
        C y2{in[(offset + 2 * stride) & mask]};
        C y3{in[(offset + 3 * stride) & mask]};
        C y4{in[(offset + 4 * stride) & mask]};
        C y5{in[(offset + 5 * stride) & mask]};
        C y6{in[(offset + 6 * stride) & mask]};
        C y7{in[(offset + 7 * stride) & mask]};
        // Calculate: data[i] += y1 + y2 + y3;
        out[0] = y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7;

        // data[i+1] = y0 + w*y1 + w^2*y2 + w^3*y3 + w^4*y4 + w^5*y5 + w^6*y6 +
        // w^7*y7;
        out[1] = y0 + rot45<T, C, F>(y1) + rot90<C, F>(y2) +
                 rot135<T, C, F>(y3) - y4 - rot45<T, C, F>(y5) -
                 rot90<C, F>(y6) - rot135<T, C, F>(y7);

        // data[i+2] = y0 + w^2*y1 + w^4*y2 + w^6*y3 + y4 + w^2*y5 + w^4*y6 +
        // w^6*y7;
        out[2] = y0 + rot90<C, F>(y1) - y2 - rot90<C, F>(y3) + y4 +
                 rot90<C, F>(y5) - y6 - rot90<C, F>(y7);

        // data[i+3] = y0 + w^3*y1 + w^6*y2 + w^9*y3 + w^12*y4 + w^15*y5 +
        // w^18*y6
        // + w^21*y7; data[i+3] = y0 + w^3*y1 + w^6*y2 + w^1*y3 + w^4*y4 +
        // w^7*y5
        // + w^2*y6 + w^5*y7;
        out[3] = y0 + rot135<T, C, F>(y1) - rot90<C, F>(y2) +
                 rot45<T, C, F>(y3) - y4 - rot135<T, C, F>(y5) +
                 rot90<C, F>(y6) - rot45<T, C, F>(y7);

        // data[i+4] = y0 + w^4*y1 + w^8*y2 + w^12*y3 + w^16*y4 + w^20*y5 +
        // w^24*y6 + w^28*y7; data[i+4] = y0 + w^4*y1 + y2 + w^4*y3 + y4 +
        // w^4*y5
        // + y6 + w^4*y7;
        out[4] = y0 - y1 + y2 - y3 + y4 - y5 + y6 - y7;

        // data[i+5] = y0 + w^5*y1 + w^10*y2 + w^15*y3 + w^20*y4 + w^25*y5 +
        // w^30*y6 + w^35*y7; data[i+5] = y0 + w^5*y1 + w^2*y2 + w^7*y3 + w^4*y4
        // + w^1*y5 + w^6*y6 + w^3*y7;
        out[5] = y0 - rot45<T, C, F>(y1) + rot90<C, F>(y2) -
                 rot135<T, C, F>(y3) - y4 + rot45<T, C, F>(y5) -
                 rot90<C, F>(y6) + rot135<T, C, F>(y7);

        // data[i+6] = y0 + w^6*y1 + w^12*y2 + w^18*y3 + w^24*y4 + w^30*y5 +
        // w^36*y6 + w^42*y7; data[i+6] = y0 + w^6*y1 + w^4*y2 + w^2*y3 + y4 +
        // w^6*y5 + w^4*y6 + w^2*y7;
        out[6] = y0 - rot90<C, F>(y1) - y2 + rot90<C, F>(y3) + y4 -
                 rot90<C, F>(y5) - y6 + rot90<C, F>(y7);

        // data[i+7] = y0 + w^7*y1 + w^14*y2 + w^21*y3 + w^28*y4 + w^35*y5 +
        // w^42*y6 + w^49*y7; data[i+7] = y0 + w^7*y1 + w^6*y2 + w^5*y3 + w^4*y4
        // + w^3*y5 + w^2*y6 + w^1*y7;
        out[7] = y0 - rot135<T, C, F>(y1) - rot90<C, F>(y2) -
                 rot45<T, C, F>(y3) - y4 + rot135<T, C, F>(y5) +
                 rot90<C, F>(y6) + rot45<T, C, F>(y7);
        break;
    }
    default: {
        transformRecursion<T, F>(in, out, twiddle, offset, 2 * stride, N / 2,
                                 mask);
        transformRecursion<T, F>(in, out + N / 2, twiddle, offset + stride,
                                 4 * stride, N / 4, mask);
        transformRecursion<T, F>(in, out + 3 * N / 4, twiddle, offset - stride,
                                 4 * stride, N / 4, mask);
        C u1, u3, z1, z3;
        for (std::size_t i = 0; i < N / 4; i++) {
            u1 = out[i];
            u3 = out[i + N / 4];
            z1 = out[i + N / 2] * twiddle[i * stride];
            z3 = out[i + 3 * N / 4] * std::conj(twiddle[i * stride]);

            // Calculate: data[i*S*2] = u1 + z1 + z3
            out[i] = u1 + z1 + z3;

            // Calculate: data[i*S*2 + N/2] = u1 - z1 - z3
            out[i + N / 2] = u1 - z1 - z3;

            // Calculate: data[i*S*2 + N/4] = u3 -j*(z1 - z3)
            out[i + N / 4] = u3 + rot90<C, F>(z1 - z3);

            // Calculate: data[i*S*2 + 3*N/4] = u3 + j*(z1 - z3)
            out[i + 3 * N / 4] = u3 - rot90<C, F>(z1 - z3);
        }
    }
    }
}

template <typename T, bool F = false>
void cfftForward(const std::complex<T>* in, std::complex<T>* out,
                 const std::complex<T>* twiddle, std::size_t nfft)
{
    transformRecursion<T, F>(in, out, twiddle, (std::size_t)0, (std::size_t)1,
                             nfft, nfft - 1);
}

template <typename T, bool F = true>
void cfftInverse(const std::complex<T>* in, std::complex<T>* out,
                 const std::complex<T>* twiddle, std::size_t nfft)
{
    transformRecursion<T, F>(in, out, twiddle, 0, 1, nfft, nfft - 1);
}

template <typename T>
void populateRfftTwiddles(std::complex<T>* twiddleFactors,
                          const std::size_t nfft, bool inverseTransform)
{
    // Populate twiddle factors for the real-valued fft. I.e. the first half
    // should only contain the even numbered entries of the twiddle factors
    // corresponding to a FFT of nfft/2. The second half should be the
    // corresponding odd entries.
    T pi{std::acos((T)-1)};
    for (std::size_t idx = 0; idx < nfft / 2; idx++) {
        twiddleFactors[idx] =
            inverseTransform ? std::exp(std::complex<T>(0, (T)(2 * idx) * (T)2 *
                                                               pi / ((T)nfft)))
                             : std::exp(std::complex<T>(
                                   0, -(T)(2 * idx) * (T)2 * pi / ((T)nfft)));
    }
    for (std::size_t idx = nfft / 2; idx < nfft; idx++) {
        twiddleFactors[idx] =
            inverseTransform
                ? std::exp(std::complex<T>(0, (T)(2 * idx + 1) * (T)2 * pi /
                                                  ((T)nfft)))
                : std::exp(std::complex<T>(0, -(T)(2 * idx + 1) * (T)2 * pi /
                                                  ((T)nfft)));
    }
}

template <typename T>
void interleaveSequence(const T* in, std::complex<T>* out,
                        const std::size_t nfft)
{
    // Interleave the real input sequence with even samples as the real values
    // and odd samples as the imaginary values of the complex output array.
    for (std::size_t idx = 0; idx < nfft / 2; idx++) {
        out[idx].real(in[idx * 2]);
        out[idx].imag(in[idx * 2 + 1]);
    }
}

template <typename T>
void deinterleaveSequence(const std::complex<T>* in, T* out,
                          const std::size_t nfft)
{
    for (std::size_t idx = 0; idx < nfft / 2; idx++) {
        out[2 * idx] = in[idx].real();
        out[2 * idx + 1] = in[idx].imag();
    }
}

template <typename T>
void rfftForward(std::complex<T>* in, std::complex<T>* out,
                 const std::complex<T>* twiddleFactors, const std::size_t nfft)
{
    using C = std::complex<T>;
    // Perform the fft on two real sequences (encoded using the real and
    // imaginary values of the input array) simultaneously.
    cfftForward<T>(in, out, twiddleFactors, nfft / 2);

    // Unscramble the intermediate spectrum into the symmetric half-spectrum
    // Tmp variables
    C xEven, xOdd, xEvenInv, xOddInv;
    C j{0, 1};
    // Handle the i=0 and Nyquist case seperately
    out[0] = std::conj(out[0]) + j * std::conj(out[0]);
    out[nfft / 2].real(out[0].imag());
    out[nfft / 2].imag(0);
    out[0].imag(0);

    for (unsigned int idx = 1; idx < nfft / 4; ++idx) {
        xEven = T(0.5) * (out[idx] + std::conj(out[nfft / 2 - idx]));
        xOdd = -T(0.5) * j * (out[idx] - std::conj(out[nfft / 2 - idx]));
        xEvenInv = T(0.5) * (out[nfft / 2 - idx] + std::conj(out[idx]));
        xOddInv = -T(0.5) * j * (out[nfft / 2 - idx] - std::conj(out[idx]));
        // Even / Odd entry lookup of the twiddle factors. This is done to reuse
        // the twiddle factors for the cfft of size nfft/2.
        if (idx % 2 == 0) {
            out[idx] = xEven + xOdd * twiddleFactors[idx / 2];
            out[nfft / 2 - idx] =
                xEvenInv + xOddInv * twiddleFactors[nfft / 4 - (idx / 2)];
        } else {
            out[idx] = xEven + xOdd * twiddleFactors[nfft / 2 + idx / 2];
            out[nfft / 2 - idx] =
                xEvenInv +
                xOddInv * twiddleFactors[nfft / 2 + nfft / 4 - 1 - (idx / 2)];
        }
    }
    xEven = T(0.5) * (out[nfft / 4] + std::conj(out[nfft / 4]));
    xOdd = -T(0.5) * j * (out[nfft / 4] - std::conj(out[nfft / 4]));
    out[nfft / 4] = xEven + xOdd * twiddleFactors[nfft / 8];
}

template <typename T>
void rfftForward(const T* inputRealSequence,
                 std::complex<T>* complexInterleavedScratch,
                 std::complex<T>* outputHalfSpectrum,
                 const std::complex<T>* twiddleFactors, const std::size_t nfft)
{
    interleaveSequence<T>(inputRealSequence, complexInterleavedScratch, nfft);
    rfftForward<T>(complexInterleavedScratch, outputHalfSpectrum,
                   twiddleFactors, nfft);
}

template <typename T>
void rfftInverse(const std::complex<T>* in, std::complex<T>* scratch,
                 std::complex<T>* out, const std::complex<T>* twiddleFactors,
                 std::size_t nfft)
{
    // Note that this function will overwrite the input!
    using C = std::complex<T>;
    // Tmp variables
    C xEven, xOdd, xEvenInv, xOddInv;
    C j{0, 1};

    scratch[0].imag(in[nfft / 2].real());
    // Handle i = 0 seperately because the Nyquist sample is
    // stored as spectrum[0].imag().
    xEven.real(T(0.5) * (in[0].real() + scratch[0].imag()));
    xEven.imag(0);
    xOdd.real(T(0.5) * (in[0].real() - scratch[0].imag()));
    xOdd.imag(0);
    scratch[0] = xEven + j * xOdd;
    for (std::size_t idx = 1; idx < nfft / 4; ++idx) {
        xEven = T(0.5) * (in[idx] + std::conj(in[nfft / 2 - idx]));
        xOdd = T(0.5) * j * (in[idx] - std::conj(in[nfft / 2 - idx]));
        xEvenInv = T(0.5) * (in[nfft / 2 - idx] + std::conj(in[idx]));
        xOddInv = T(0.5) * j * (in[nfft / 2 - idx] - std::conj(in[idx]));
        // Even / Odd entry lookup of the twiddle factors. This is done to reuse
        // the twiddle factors for the cfft of size nfft/2.
        if (idx % 2 == 0) {
            scratch[idx] = xEven + xOdd * twiddleFactors[idx / 2];
            scratch[nfft / 2 - idx] =
                xEvenInv + xOddInv * twiddleFactors[nfft / 4 - (idx / 2)];
        } else {
            scratch[idx] = xEven + xOdd * twiddleFactors[nfft / 2 + idx / 2];
            scratch[nfft / 2 - idx] =
                xEvenInv +
                xOddInv * twiddleFactors[nfft / 2 + nfft / 4 - 1 - (idx / 2)];
        }
    }
    xEven = T(0.5) * (in[nfft / 4] + std::conj(in[nfft / 4]));
    xOdd = T(0.5) * j * (in[nfft / 4] - std::conj(in[nfft / 4]));
    scratch[nfft / 4] = xEven + xOdd * twiddleFactors[nfft / 8];

    // Perform a complex valued FFT of the half-length complex sequence
    cfftInverse(scratch, out, twiddleFactors, nfft / 2);
}

template <typename T>
void rfftInverse(const std::complex<T>* inputHalfSpectrum,
                 std::complex<T>* complexInterleavedScratchInput,
                 std::complex<T>* complexInterleavedScratchOutput,
                 T* outputRealSequence, const std::complex<T>* twiddleFactors,
                 const std::size_t nfft)
{
    // Note that the inputHalfSpectrum will be overwritten!
    rfftInverse<T>(inputHalfSpectrum, complexInterleavedScratchInput,
                   complexInterleavedScratchOutput, twiddleFactors, nfft);
    deinterleaveSequence<T>(complexInterleavedScratchOutput, outputRealSequence,
                            nfft);
    for (size_t idx = 0; idx < nfft; idx++) {
        outputRealSequence[idx] /= (T)(nfft / 2);
    }
}
} // namespace internal
} // namespace fft
