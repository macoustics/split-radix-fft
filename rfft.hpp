#include "cfft.hpp"

namespace fft {
template <typename T>
void populateRfftTwiddles(std::complex<T>* twiddleFactors, const std::size_t nfft, bool inverseTransform){
    // Populate twiddle factors for the real-valued fft. I.e. the first half
    // should only contain the even numbered entries of the twiddle factors
    // corresponding to a FFT of nfft/2. The second half should be the corresponding 
    // odd entries.
    T pi{std::acos((T)-1)};
    for (std::size_t idx = 0; idx < nfft/2; idx++) {
    twiddleFactors[idx] =
        inverseTransform ? std::exp(std::complex<T>(0, (T)(2*idx) * (T)2 * pi / ((T)nfft)))
                        : std::exp(std::complex<T>(0, -(T)(2*idx) * (T)2 * pi / ((T)nfft)));
    }
    for (std::size_t idx = nfft/2; idx < nfft; idx++) {
    twiddleFactors[idx] =
        inverseTransform ? std::exp(std::complex<T>(0, (T)(2*idx+1) * (T)2 * pi / ((T)nfft)))
                        : std::exp(std::complex<T>(0, -(T)(2*idx+1) * (T)2 * pi / ((T)nfft)));
    }
}

template <typename T>
void interleaveSequence(const T* in, std::complex<T>* out, const std::size_t nfft){
    // Interleave the real input sequence with even samples as the real values and 
    // odd samples as the imaginary values of the complex output array.
    for(std::size_t idx = 0; idx<nfft/2; idx++){
        out[idx].real(in[idx*2]);
        out[idx].imag(in[idx*2+1]);
    }
}

template <typename T>
void deinterleaveSequence(const std::complex<T>* in, T* out, const std::size_t nfft){
    for(std::size_t idx = 0; idx<nfft/2; idx++){
        out[2*idx] = in[idx].real();
        out[2*idx+1] = in[idx].imag();
    }
}

template <typename T>
void rfftForward(std::complex<T>* in, std::complex<T>* out, const std::complex<T>* twiddleFactors, const std::size_t nfft){
    using C = std::complex<T>;
    // Perform the fft on two real sequences (encoded using the real and imaginary values of the input array) simultaneously.
    cfftForward<T>(in, out, twiddleFactors, nfft/2);

    // Unscramble the intermediate spectrum into the symmetric half-spectrum
    // Tmp variables
    C xEven, xOdd, xEvenInv, xOddInv;
    C j {0,1};
    // Handle the i=0 and Nyquist case seperately
    out[0] = std::conj(out[0]) + j*std::conj(out[0]);
    out[nfft/2].real(out[0].imag());
    out[nfft/2].imag(0);
    out[0].imag(0);
    
    for (unsigned int idx = 1; idx < nfft/4; ++idx) {
        xEven = T(0.5) * (out[idx] + std::conj(out[nfft/2 - idx]));
        xOdd = -T(0.5) * j * (out[idx] - std::conj(out[nfft/2 - idx]));
        xEvenInv = T(0.5) * (out[nfft/2 - idx] + std::conj(out[idx]));
        xOddInv = -T(0.5) * j * (out[nfft/2 - idx] - std::conj(out[idx]));
        // Even / Odd entry lookup of the twiddle factors. This is done to reuse
        // the twiddle factors for the cfft of size nfft/2.
        if(idx%2 == 0){
            out[idx] = xEven + xOdd * twiddleFactors[idx/2];
            out[nfft/2-idx] = xEvenInv + xOddInv * twiddleFactors[nfft/4-(idx/2)];
        } else{
            out[idx] = xEven + xOdd * twiddleFactors[nfft/2 + idx/2];
            out[nfft/2-idx] = xEvenInv + xOddInv * twiddleFactors[nfft/2+nfft/4-1-(idx/2)];
        }
    }
    xEven = T(0.5) * (out[nfft/4] + std::conj(out[nfft/4]));
    xOdd = -T(0.5) * j * (out[nfft/4] - std::conj(out[nfft/4]));
    out[nfft/4] = xEven + xOdd * twiddleFactors[nfft/8];
    
}

template <typename T>
void rfftForward(const T* inputRealSequence, std::complex<T>* complexInterleavedScratch, std::complex<T>* outputHalfSpectrum, const std::complex<T>* twiddleFactors, const std::size_t nfft){
    interleaveSequence<T>(inputRealSequence, complexInterleavedScratch, nfft);
    rfftForward<T>(complexInterleavedScratch, outputHalfSpectrum, twiddleFactors, nfft);
}


template <typename T>
void rfftInverse(std::complex<T>* in, std::complex<T>* out, const std::complex<T>* twiddleFactors, std::size_t nfft){
    // Note that this function will overwrite the input!
    using C = std::complex<T>;
    // Tmp variables
    C xEven, xOdd, xEvenInv, xOddInv;
    C j {0,1};

    in[0].imag(in[nfft/2].real());
    // Handle i = 0 seperately because the Nyquist sample is
    // stored as spectrum[0].imag().
    xEven.real(T(0.5) * (in[0].real() + in[0].imag()));
    xEven.imag(0);
    xOdd.real(T(0.5) * (in[0].real() - in[0].imag()));
    xOdd.imag(0);
    in[0] = xEven + j * xOdd;
    for (std::size_t idx = 1; idx < nfft/4; ++idx) {
        xEven = T(0.5) * (in[idx] + std::conj(in[nfft/2 - idx]));
        xOdd = T(0.5) * j * (in[idx] - std::conj(in[nfft/2 - idx]));
        xEvenInv = T(0.5) * (in[nfft/2-idx] + std::conj(in[idx]));
        xOddInv = T(0.5) * j * (in[nfft/2-idx] - std::conj(in[idx]));
        // Even / Odd entry lookup of the twiddle factors. This is done to reuse
        // the twiddle factors for the cfft of size nfft/2.
        if(idx%2 == 0){
            in[idx] = xEven + xOdd * twiddleFactors[idx/2];
            in[nfft/2-idx] = xEvenInv + xOddInv * twiddleFactors[nfft/4-(idx/2)];
        } else{
            in[idx] = xEven + xOdd * twiddleFactors[nfft/2 + idx/2];
            in[nfft/2-idx] = xEvenInv + xOddInv * twiddleFactors[nfft/2+nfft/4-1-(idx/2)];
        }
    }
    xEven = T(0.5) * (in[nfft/4] + std::conj(in[nfft/4]));
    xOdd = T(0.5) * j * (in[nfft/4] - std::conj(in[nfft/4]));
    in[nfft/4] = xEven + xOdd * twiddleFactors[nfft/8];

    // Perform a complex valued FFT of the half-length complex sequence
    cfftInverse(in, out, twiddleFactors, nfft/2);
}

template <typename T>
void rfftInverse(std::complex<T>* inputHalfSpectrum, std::complex<T>* complexInterleavedScratch, T* outputRealSequence, const std::complex<T>* twiddleFactors, const std::size_t nfft){
    // Note that the inputHalfSpectrum will be overwritten!
    rfftInverse<T>(inputHalfSpectrum, complexInterleavedScratch, twiddleFactors, nfft);
    deinterleaveSequence<T>(complexInterleavedScratch, outputRealSequence, nfft);
    for(size_t idx = 0; idx < nfft; idx++){
        outputRealSequence[idx] /= (T)(nfft/2);
    }
}
} // namespace fft