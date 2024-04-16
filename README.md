# split-radix-fft
A small lib for computing the real or complex fft for 1D arrays of radix2 length

This repo contains code for implementing the split radix conjugate pair fft algorithm. The code does not allocate memory, hence, the user is required to pass the required data-structures to the functions. The functions are stand-alone and can be used as is or wrapped in a class if that is desired.

This fft is not particularly fast and just provide base-cases for size 1, 2, 4, and 8.

## Options:
- performCfftForward: Perform the fft assuming complex valued input sequence.
- performCfftBackward: Perform the inverse fft assuming complex valued output sequence. Note that this inverse is not normalized. If a normalized inverse is desired, divide the result by the length of the sequence.
- populateCfftTwiddleFactorsForward: Calculates the twiddle factors for the forward cfft transforms.
- populateCfftTwiddleFactorsBackward: Calculates the twiddle factors for the backward cfft transform.

- performRfftForward: Perform the fft assuming real-valued input sequence. The output is the first half of the symmetric half-spectrum of size = nfft/2 + 1, where the value at DC and Nyquist have zero imaginary components.
- performRfftBackward: Perform the inverse fft assuming a real-valued output sequence. If this option is only provided with a single scratch space this operation overwrites the input half-spectrum an intermediate result.
- populateRfftTwiddleFactorsForward: Calculates the twiddle factors for the forward rfft transform.
- populateRfftTwiddleFactorsBackward: Calculates the twiddle factors for the backward rfft transform.



## Known issues:
 - If a single scratch-space is provided for performRfftBackward(), the function will use the input spectrum as the temporary scratch space, i.e., the input spectrum will be overwritten!

## Development:
In order to further develop the `split-radix-fft`, we provide a docker build environment. To build the container:
```bash
./build.sh -d
```
To build the code:
```bash
./build.sh
```
To run the tests:
```bash
./build.sh -t 
```


