# splitRadixFft
A small lib for computing the real or complex fft for 1D arrays of radix2 length

This repo contains code for implementing the split radix conjugate pair fft algorithm. The code does not allocate memory, hence, the user is required to pass the required data-structures to the functions. The functions are stand-alone and can be used as is or wrapped in a class if that is desired.

This fft is not particularly fast and just provide base-cases for size 1, 2, 4, and 8.

# options:
- cfftForward: Perform the fft assuming complex valued input sequence.
- cfftInverse: Perform the inverse fft assuming complex valued output sequence. Note that this inverse is not normalized. If a normalized inverse is desired, divide the result by the length of the sequence.
- populateCfftTwiddles: Calculates the twiddle factors for the cfft transforms.

- rfftForward: Perform the fft assuming real-valued input sequence. The output is the first half of the symmetric half-spectrum of size = nfft/2 + 1, where the value at DC and Nyquist have zero imaginary components.
- rfftInverse: Perform the inverse fft assuming a real-valued output sequence. This takes a half-spectrum as input and overwrites it as an intermediate result.
- populateRfftTwiddles: Calculates the twiddle factors for the rfft transforms.



# Known issues:
 - The code was originally intended for std::array and then converted to pointer-based inputs, without type checking. As such, the code is unsafe and will segfault if the size of the input arguments is too small.
 - The rfftInverse() function will use the input spectrum as the temporary scratch space, i.e., the input spectrum will be overwritten!
