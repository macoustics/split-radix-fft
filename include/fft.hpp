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
 * fft.hpp
 * Simple library for an fft. By composition of the algorithm, the rfft is by
 * default normalized but the cfft is currently not normalized.
 *
 * ==============================================================================
 */

#pragma once
#include "internal.hpp"
#include <complex>

namespace fft {
/*
 * ==============================================================================
 *
 * Public API
 * We split the library in an internal part and an external part in order to
 * separate the implementation from the interface.
 *
 * ==============================================================================
 */
enum class FFTSTATUS {
    OK = 0,
    INVALID_SIZE = -1,
    NULL_POINTER = -2,
};

template <typename T>
FFTSTATUS
populateCfftTwiddleFactorsForward(const std::size_t nfft,
                                  std::complex<T>* twiddleFactors,
                                  const std::size_t twiddleFactorSize) noexcept
{
    if (nfft != twiddleFactorSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactors == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    internal::populateCfftTwiddles<T>(twiddleFactors, nfft, false);

    return FFTSTATUS::OK;
}

template <typename T>
FFTSTATUS
populateCfftTwiddleFactorsBackward(const std::size_t nfft,
                                   std::complex<T>* twiddleFactors,
                                   const std::size_t twiddleFactorSize) noexcept
{
    if (nfft != twiddleFactorSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactors == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    internal::populateCfftTwiddles<T>(twiddleFactors, nfft, true);

    return FFTSTATUS::OK;
}

template <typename T>
FFTSTATUS
populateRfftTwiddleFactorsForward(const std::size_t nfft,
                                  std::complex<T>* twiddleFactors,
                                  const std::size_t twiddleFactorSize) noexcept
{
    if (nfft % 2 != 0) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactorSize != nfft) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactors == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    internal::populateRfftTwiddles<T>(twiddleFactors, nfft, false);

    return FFTSTATUS::OK;
}

template <typename T>
FFTSTATUS
populateRfftTwiddleFactorsBackward(const std::size_t nfft,
                                   std::complex<T>* twiddleFactors,
                                   const std::size_t twiddleFactorSize) noexcept
{
    if (nfft % 2 != 0) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactorSize != nfft) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactors == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    internal::populateRfftTwiddles<T>(twiddleFactors, nfft, true);

    return FFTSTATUS::OK;
}

template <typename T>
FFTSTATUS performCfftForward(const std::size_t nfft,
                             std::complex<T>* twiddleFactors,
                             const std::size_t twiddleFactorSize,
                             std::complex<T>* in, const std::size_t inSize,
                             std::complex<T>* out, const std::size_t outSize)
{
    if (nfft != twiddleFactorSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (nfft != inSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (nfft != outSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactors == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (in == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (out == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    internal::cfftForward<T>(in, out, twiddleFactors, nfft);

    return FFTSTATUS::OK;
}

template <typename T>
FFTSTATUS performCfftBackward(const std::size_t nfft,
                              std::complex<T>* twiddleFactors,
                              const std::size_t twiddleFactorSize,
                              std::complex<T>* in, const std::size_t inSize,
                              std::complex<T>* out, const std::size_t outSize)
{
    if (nfft != twiddleFactorSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (nfft != inSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (nfft != outSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactors == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (in == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (out == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    internal::cfftInverse<T>(in, out, twiddleFactors, nfft);

    return FFTSTATUS::OK;
}

template <typename T>
FFTSTATUS performRfftForward(const std::size_t nfft,
                             std::complex<T>* twiddleFactors,
                             const std::size_t twiddleFactorSize, const T* in,
                             const std::size_t inSize, std::complex<T>* out,
                             const std::size_t outSize,
                             std::complex<T>* scratch, std::size_t scratchSize)
{
    if (nfft % 2 != 0) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (nfft != twiddleFactorSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (nfft != inSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (outSize != (nfft / 2 + 1)) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (scratchSize != (nfft / 2 + 1)) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactors == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (in == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (out == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (scratch == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    internal::rfftForward<T>(in, scratch, out, twiddleFactors, nfft);

    return FFTSTATUS::OK;
}

template <typename T>
FFTSTATUS
performRfftBackward(const std::size_t nfft, std::complex<T>* twiddleFactors,
                    const std::size_t twiddleFactorSize,
                    const std::complex<T>* in, const std::size_t inSize, T* out,
                    const std::size_t outSize, std::complex<T>* scratch0,
                    std::complex<T>* scratch1, std::size_t scratchSize)
{
    if (nfft != twiddleFactorSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if ((nfft / 2 + 1) != inSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (nfft != outSize) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (scratchSize != (nfft / 2 + 1)) {
        return FFTSTATUS::INVALID_SIZE;
    }

    if (twiddleFactors == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (in == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (out == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (scratch0 == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    if (scratch1 == nullptr) {
        return FFTSTATUS::NULL_POINTER;
    }

    internal::rfftInverse<T>(in, scratch0, scratch1, out, twiddleFactors, nfft);

    return FFTSTATUS::OK;
}

template <typename T>
FFTSTATUS performRfftBackwardWithInputAsScratch(
    const std::size_t nfft, std::complex<T>* twiddleFactors,
    const std::size_t twiddleFactorSize, std::complex<T>* in,
    const std::size_t inSize, T* out, const std::size_t outSize,
    std::complex<T>* scratch, std::size_t scratchSize)
{
    FFTSTATUS status;
    status =
        performRfftBackward<T>(nfft, twiddleFactors, twiddleFactorSize, in,
                               inSize, out, outSize, in, scratch, scratchSize);

    return status;
}
} // namespace fft
