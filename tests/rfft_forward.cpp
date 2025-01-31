#include "splitradixfft.hpp"
#include <catch2/catch_test_macros.hpp>
#include <memory>

TEST_CASE("performRfftForwardFloat::Valid", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<float[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    in[0] = -0.4909053355714212;
    in[1] = -0.648677632197376;
    in[2] = -0.08127662969009877;
    in[3] = -0.009879307054779199;
    in[4] = 0.366704052477839;
    in[5] = -0.21992885717017768;
    in[6] = -0.18157788121511045;
    in[7] = 1.1637091789773049;
    in[8] = 0.01842903085229932;
    in[9] = 0.04797944141650834;
    in[10] = -0.43232759942875965;
    in[11] = -0.07528208264741829;
    in[12] = 0.4327277580835922;
    in[13] = 0.16114135912892974;
    in[14] = -1.2666864253030767;
    in[15] = -0.612769159796717;

    err = splitradixfft::performRfftForward<float>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize, out.get(), outSize,
                                         scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t refSize = nfft / 2 + 1;
    auto ref = std::make_unique<std::complex<float>[]>(outSize);

    ref[0] = std::complex<float>(-1.8286200891384616, 0.0);
    ref[1] = std::complex<float>(-3.1424128266660656, -1.0710854460299988);
    ref[2] = std::complex<float>(-1.2053052211709034, -0.10167973202230995);
    ref[3] = std::complex<float>(-1.3491924851640964, -2.1999673020454855);
    ref[4] = std::complex<float>(2.288824041479355, 1.125264318300506);
    ref[5] = std::complex<float>(1.3686379295241395, -0.03688362864921224);
    ref[6] = std::complex<float>(-1.3385110093902028, 1.7676404227763478);
    ref[7] = std::complex<float>(1.08562991661114, 0.8279034049432613);
    ref[8] = std::complex<float>(-1.4412059704510114, 0.0);


    for (std::size_t i = 0; i < outSize; i++) {
        REQUIRE(std::fabs(ref[i] - out[i]) < 1e-5f);
    }
}

TEST_CASE("performRfftForwardFloat::InvalidTwiddleSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<float[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = splitradixfft::performRfftForward<float>(nfft, twiddleFactors.get(), nfft - 1,
                                         in.get(), inSize, out.get(), outSize,
                                         scratch.get(), scratchSize);

    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftForwardFloat::InvalidInputSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<float[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = splitradixfft::performRfftForward<float>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize - 1, out.get(),
                                         outSize, scratch.get(), scratchSize);

    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftForwardFloat::InvalidOutputSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<float[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = splitradixfft::performRfftForward<float>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize, out.get(),
        outSize - 1, scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftForwardFloat::NullTwiddle", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<float[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = splitradixfft::performRfftForward<float>(nfft, nullptr, nfft, in.get(), inSize,
                                         out.get(), outSize, scratch.get(),
                                         scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftForwardFloat::NullInput", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<float[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = splitradixfft::performRfftForward<float>(nfft, twiddleFactors.get(), nfft,
                                         nullptr, inSize, out.get(), outSize,
                                         scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftForwardFloat::NullOutput", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<float[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<float>[]>(scratchSize);

    err = splitradixfft::performRfftForward<float>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize, nullptr, outSize,
                                         scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftForwardDouble::Valid", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<double[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    in[0] = -0.4909053355714212;
    in[1] = -0.648677632197376;
    in[2] = -0.08127662969009877;
    in[3] = -0.009879307054779199;
    in[4] = 0.366704052477839;
    in[5] = -0.21992885717017768;
    in[6] = -0.18157788121511045;
    in[7] = 1.1637091789773049;
    in[8] = 0.01842903085229932;
    in[9] = 0.04797944141650834;
    in[10] = -0.43232759942875965;
    in[11] = -0.07528208264741829;
    in[12] = 0.4327277580835922;
    in[13] = 0.16114135912892974;
    in[14] = -1.2666864253030767;
    in[15] = -0.612769159796717;

    err = splitradixfft::performRfftForward<double>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize, out.get(), outSize,
                                         scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t refSize = nfft / 2 + 1;
    auto ref = std::make_unique<std::complex<double>[]>(outSize);

    ref[0] = std::complex<double>(-1.8286200891384616, 0.0);
    ref[1] = std::complex<double>(-3.1424128266660656, -1.0710854460299988);
    ref[2] = std::complex<double>(-1.2053052211709034, -0.10167973202230995);
    ref[3] = std::complex<double>(-1.3491924851640964, -2.1999673020454855);
    ref[4] = std::complex<double>(2.288824041479355, 1.125264318300506);
    ref[5] = std::complex<double>(1.3686379295241395, -0.03688362864921224);
    ref[6] = std::complex<double>(-1.3385110093902028, 1.7676404227763478);
    ref[7] = std::complex<double>(1.08562991661114, 0.8279034049432613);
    ref[8] = std::complex<double>(-1.4412059704510114, 0.0);

    for (std::size_t i = 0; i < outSize; i++) {
        REQUIRE(std::fabs(ref[i] - out[i]) < 1e-5f);
    }
}

TEST_CASE("performRfftForwardDouble::InvalidTwiddleSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<double[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = splitradixfft::performRfftForward<double>(nfft, twiddleFactors.get(), nfft - 1,
                                          in.get(), inSize, out.get(), outSize,
                                          scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftForwardDouble::InvalidInputSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<double[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = splitradixfft::performRfftForward<double>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize - 1, out.get(),
                                          outSize, scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftForwardDouble::InvalidOutputSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<double[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = splitradixfft::performRfftForward<double>(
        nfft, twiddleFactors.get(), nfft, in.get(), inSize, out.get(),
        outSize - 1, scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performRfftForwardDouble::NullTwiddle", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<double[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = splitradixfft::performRfftForward<double>(nfft, nullptr, nfft, in.get(), inSize,
                                          out.get(), outSize, scratch.get(),
                                          scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftForwardDouble::NullInput", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<double[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = splitradixfft::performRfftForward<double>(nfft, twiddleFactors.get(), nfft,
                                          nullptr, inSize, out.get(), outSize,
                                          scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performRfftForwardDouble::NullOutput", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateRfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<double[]>(inSize);

    const std::size_t outSize = nfft / 2 + 1;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    const std::size_t scratchSize = nfft / 2 + 1;
    auto scratch = std::make_unique<std::complex<double>[]>(scratchSize);

    err = splitradixfft::performRfftForward<double>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize, nullptr, outSize,
                                          scratch.get(), scratchSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}
