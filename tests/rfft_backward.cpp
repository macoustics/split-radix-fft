#include "fft.hpp"
#include <catch2/catch_test_macros.hpp>
#include <memory>

TEST_CASE("performRfftBackwardFloat::Valid", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<float[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    in[0] = std::complex<float>(-1.8286200891384616, 0.0);
    in[1] = std::complex<float>(-3.1424128266660656, -1.0710854460299988);
    in[2] = std::complex<float>(-1.2053052211709034, -0.10167973202230995);
    in[3] = std::complex<float>(-1.3491924851640964, -2.1999673020454855);
    in[4] = std::complex<float>(2.288824041479355, 1.125264318300506);
    in[5] = std::complex<float>(1.3686379295241395, -0.03688362864921224);
    in[6] = std::complex<float>(-1.3385110093902028, 1.7676404227763478);
    in[7] = std::complex<float>(1.08562991661114, 0.8279034049432613);
    in[8] = std::complex<float>(-1.4412059704510114, 0.0);

    err = fft::performRfftBackwardWithInputAsScratch<float>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize, out.get(), outSize,
        scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t refSize = nfft / 2 + 1;
    auto ref = std::make_unique<float[]>(outSize);

    ref[0] = -0.4909053355714212;
    ref[1] = -0.648677632197376;
    ref[2] = -0.08127662969009877;
    ref[3] = -0.009879307054779199;
    ref[4] = 0.366704052477839;
    ref[5] = -0.21992885717017768;
    ref[6] = -0.18157788121511045;
    ref[7] = 1.1637091789773049;
    ref[8] = 0.01842903085229932;
    ref[9] = 0.04797944141650834;
    ref[10] = -0.43232759942875965;
    ref[11] = -0.07528208264741829;
    ref[12] = 0.4327277580835922;
    ref[13] = 0.16114135912892974;
    ref[14] = -1.2666864253030767;
    ref[15] = -0.612769159796717;

    for (std::size_t i = 0; i < outSize; i++) {
        REQUIRE(std::fabs(ref[i] - out[i]) < 1e-5f);
    }
}

TEST_CASE("performRfftBackwardFloat::InvalidTwiddleSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<float[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<float>(
        nfft, twiddleFactors.get(), nfft - 1, in.get(), inSize, out.get(),
        outSize, scratch.get(), scratchSize);

    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftBackwardFloat::InvalidInputSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<float[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<float>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize - 1, out.get(),
        outSize, scratch.get(), scratchSize);

    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftBackwardFloat::InvalidOutputSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<float[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<float>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize, out.get(),
        outSize - 1, scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftBackwardFloat::NullTwiddle", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<float[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<float>(
        nfft, nullptr, nfft, in.get(), inSize, out.get(), outSize,
        scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftBackwardFloat::NullInput", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<float[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<float>(
        nfft, twiddleFactors.get(), nfft, nullptr, inSize, out.get(), outSize,
        scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftBackwardFloat::NullOutput", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<float[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<float>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize, nullptr, outSize,
        scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftBackwardDouble::Valid", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<double[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    in[0] = std::complex<double>(-1.8286200891384616, 0.0);
    in[1] = std::complex<double>(-3.1424128266660656, -1.0710854460299988);
    in[2] = std::complex<double>(-1.2053052211709034, -0.10167973202230995);
    in[3] = std::complex<double>(-1.3491924851640964, -2.1999673020454855);
    in[4] = std::complex<double>(2.288824041479355, 1.125264318300506);
    in[5] = std::complex<double>(1.3686379295241395, -0.03688362864921224);
    in[6] = std::complex<double>(-1.3385110093902028, 1.7676404227763478);
    in[7] = std::complex<double>(1.08562991661114, 0.8279034049432613);
    in[8] = std::complex<double>(-1.4412059704510114, 0.0);

    err = fft::performRfftBackwardWithInputAsScratch<double>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize, out.get(), outSize,
        scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t refSize = nfft / 2 + 1;
    auto ref = std::make_unique<double[]>(outSize);

    ref[0] = -0.4909053355714212;
    ref[1] = -0.648677632197376;
    ref[2] = -0.08127662969009877;
    ref[3] = -0.009879307054779199;
    ref[4] = 0.366704052477839;
    ref[5] = -0.21992885717017768;
    ref[6] = -0.18157788121511045;
    ref[7] = 1.1637091789773049;
    ref[8] = 0.01842903085229932;
    ref[9] = 0.04797944141650834;
    ref[10] = -0.43232759942875965;
    ref[11] = -0.07528208264741829;
    ref[12] = 0.4327277580835922;
    ref[13] = 0.16114135912892974;
    ref[14] = -1.2666864253030767;
    ref[15] = -0.612769159796717;

    for (std::size_t i = 0; i < outSize; i++) {
        REQUIRE(std::fabs(ref[i] - out[i]) < 1e-5f);
    }
}

TEST_CASE("performRfftBackwardDoubleDualScratch::Valid", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<double[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch0 = std::make_unique<std::complex<double>[]>(scratchSize);
    auto scratch1 = std::make_unique<std::complex<double>[]>(scratchSize);

    in[0] = std::complex<double>(-1.8286200891384616, 0.0);
    in[1] = std::complex<double>(-3.1424128266660656, -1.0710854460299988);
    in[2] = std::complex<double>(-1.2053052211709034, -0.10167973202230995);
    in[3] = std::complex<double>(-1.3491924851640964, -2.1999673020454855);
    in[4] = std::complex<double>(2.288824041479355, 1.125264318300506);
    in[5] = std::complex<double>(1.3686379295241395, -0.03688362864921224);
    in[6] = std::complex<double>(-1.3385110093902028, 1.7676404227763478);
    in[7] = std::complex<double>(1.08562991661114, 0.8279034049432613);
    in[8] = std::complex<double>(-1.4412059704510114, 0.0);

    err = fft::performRfftBackward<double>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize, out.get(), outSize,
        scratch0.get(), scratch1.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t refSize = nfft / 2 + 1;
    auto ref = std::make_unique<double[]>(outSize);

    ref[0] = -0.4909053355714212;
    ref[1] = -0.648677632197376;
    ref[2] = -0.08127662969009877;
    ref[3] = -0.009879307054779199;
    ref[4] = 0.366704052477839;
    ref[5] = -0.21992885717017768;
    ref[6] = -0.18157788121511045;
    ref[7] = 1.1637091789773049;
    ref[8] = 0.01842903085229932;
    ref[9] = 0.04797944141650834;
    ref[10] = -0.43232759942875965;
    ref[11] = -0.07528208264741829;
    ref[12] = 0.4327277580835922;
    ref[13] = 0.16114135912892974;
    ref[14] = -1.2666864253030767;
    ref[15] = -0.612769159796717;

    for (std::size_t i = 0; i < outSize; i++) {
        REQUIRE(std::fabs(ref[i] - out[i]) < 1e-5f);
    }
}

TEST_CASE("performRfftBackwardDouble::InvalidTwiddleSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<double[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<double>(
        nfft, twiddleFactors.get(), nfft - 1, in.get(), inSize, out.get(),
        outSize, scratch.get(), scratchSize);

    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftBackwardDouble::InvalidInputSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<double[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<double>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize - 1, out.get(),
        outSize, scratch.get(), scratchSize);

    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftBackwardDouble::InvalidOutputSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<double[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<double>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize, out.get(),
        outSize - 1, scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftBackwardDouble::NullTwiddle", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<double[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<double>(
        nfft, nullptr, nfft, in.get(), inSize, out.get(), outSize,
        scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftBackwardDouble::NullInput", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<double[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<double>(
        nfft, twiddleFactors.get(), nfft, nullptr, inSize, out.get(), outSize,
        scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftBackwardDouble::NullOutput", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateRfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft / 2 + 1;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<double[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = fft::performRfftBackwardWithInputAsScratch<double>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize, nullptr, outSize,
        scratch.get(), scratchSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}
