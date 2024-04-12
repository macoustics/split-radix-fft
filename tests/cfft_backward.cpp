#include "fft.hpp"
#include <catch2/catch_test_macros.hpp>
#include <memory>

TEST_CASE("performCfftBackwardFloat::Valid", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    in[0] = std::complex<float>(-1.0911257041712146, -2.9158263577231676);
    in[1] = std::complex<float>(-3.8871063587373555, -1.8534222089417363);
    in[2] = std::complex<float>(-1.7895294293147468, 2.0243532818942183);
    in[3] = std::complex<float>(1.336336765713034, 3.8316062823627943);
    in[4] = std::complex<float>(-0.30587995018617775, 3.8479477013392422);
    in[5] = std::complex<float>(-2.369579527779574, 2.1181125417824473);
    in[6] = std::complex<float>(-0.48654312248145626, 5.999469885704463);
    in[7] = std::complex<float>(3.07126264502644, -4.544087664490265);
    in[8] = std::complex<float>(-3.4117944866532133, 2.970668419018221);
    in[9] = std::complex<float>(-2.5939169789902357, 1.9395905791169208);
    in[10] = std::complex<float>(0.18798637625081538, 4.3471001073478615);
    in[11] = std::complex<float>(2.085603458970295, -5.302440146753907);
    in[12] = std::complex<float>(10.481976453272209, 2.0108749765798404);
    in[13] = std::complex<float>(4.9350060416874815, 6.821330101253683);
    in[14] = std::complex<float>(6.882567724330194, -6.0410587214513365);
    in[15] = std::complex<float>(-6.890578226657826, -0.3278280227882546);

    err = fft::performCfftBackward<float>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize, out.get(), outSize);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t refSize = nfft;
    auto ref = std::make_unique<std::complex<float>[]>(outSize);

    ref[0] = std::complex<float>(6.154685680278668, 14.926390754251024);
    ref[1] = std::complex<float>(-11.493696322988079, -31.87690289927678);
    ref[2] = std::complex<float>(-28.75392378589612, -23.743151516718967);
    ref[3] = std::complex<float>(-7.79443218235288, 6.917538087263352);
    ref[4] = std::complex<float>(-14.48966580140415, -3.9344212811526966);
    ref[5] = std::complex<float>(23.00131825360529, 23.803942307220122);
    ref[6] = std::complex<float>(7.106454905495976, 9.582260381863104);
    ref[7] = std::complex<float>(21.70282823699048, -8.45570548408146);
    ref[8] = std::complex<float>(14.780630041814149, 9.56066783116766);
    ref[9] = std::complex<float>(6.342736476721644, -35.00266455224904);
    ref[10] = std::complex<float>(-13.430194051902703, -3.8599450263544313);
    ref[11] = std::complex<float>(-5.257970425183435, 9.979797908837504);
    ref[12] = std::complex<float>(16.247055328357742, 3.102021652590557);
    ref[13] = std::complex<float>(-15.91597417644847, -23.621779576493413);
    ref[14] = std::complex<float>(-23.63840384333899, -5.195086305285825);
    ref[15] = std::complex<float>(7.980540399511438, 11.16381599484859);

    for (std::size_t i = 0; i < outSize; i++) {
        REQUIRE(std::fabs(ref[i] - out[i]) < 1e-5f);
    }
}

TEST_CASE("performCfftBackwardFloat::InvalidTwiddleSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = fft::performCfftBackward<float>(nfft, twiddleFactors.get(), nfft - 1,
                                          in.get(), inSize, out.get(), outSize);
    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftBackwardFloat::InvalidInputSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = fft::performCfftBackward<float>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize - 1, out.get(),
                                          outSize);
    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftBackwardFloat::InvalidOutputSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = fft::performCfftBackward<float>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize, out.get(),
                                          outSize - 1);
    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftBackwardFloat::NullTwiddle", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = fft::performCfftBackward<float>(nfft, nullptr, nfft, in.get(), inSize,
                                          out.get(), outSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftBackwardFloat::NullInput", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = fft::performCfftBackward<float>(nfft, twiddleFactors.get(), nfft,
                                          nullptr, inSize, out.get(), outSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftBackwardFloat::NullOutput", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = fft::performCfftBackward<float>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize, nullptr, outSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftBackwardDouble::Valid", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    in[0] = std::complex<double>(-1.0911257041712146, -2.9158263577231676);
    in[1] = std::complex<double>(-3.8871063587373555, -1.8534222089417363);
    in[2] = std::complex<double>(-1.7895294293147468, 2.0243532818942183);
    in[3] = std::complex<double>(1.336336765713034, 3.8316062823627943);
    in[4] = std::complex<double>(-0.30587995018617775, 3.8479477013392422);
    in[5] = std::complex<double>(-2.369579527779574, 2.1181125417824473);
    in[6] = std::complex<double>(-0.48654312248145626, 5.999469885704463);
    in[7] = std::complex<double>(3.07126264502644, -4.544087664490265);
    in[8] = std::complex<double>(-3.4117944866532133, 2.970668419018221);
    in[9] = std::complex<double>(-2.5939169789902357, 1.9395905791169208);
    in[10] = std::complex<double>(0.18798637625081538, 4.3471001073478615);
    in[11] = std::complex<double>(2.085603458970295, -5.302440146753907);
    in[12] = std::complex<double>(10.481976453272209, 2.0108749765798404);
    in[13] = std::complex<double>(4.9350060416874815, 6.821330101253683);
    in[14] = std::complex<double>(6.882567724330194, -6.0410587214513365);
    in[15] = std::complex<double>(-6.890578226657826, -0.3278280227882546);

    err = fft::performCfftBackward<double>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize, out.get(), outSize);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t refSize = nfft;
    auto ref = std::make_unique<std::complex<double>[]>(outSize);

    ref[0] = std::complex<double>(6.154685680278668, 14.926390754251024);
    ref[1] = std::complex<double>(-11.493696322988079, -31.87690289927678);
    ref[2] = std::complex<double>(-28.75392378589612, -23.743151516718967);
    ref[3] = std::complex<double>(-7.79443218235288, 6.917538087263352);
    ref[4] = std::complex<double>(-14.48966580140415, -3.9344212811526966);
    ref[5] = std::complex<double>(23.00131825360529, 23.803942307220122);
    ref[6] = std::complex<double>(7.106454905495976, 9.582260381863104);
    ref[7] = std::complex<double>(21.70282823699048, -8.45570548408146);
    ref[8] = std::complex<double>(14.780630041814149, 9.56066783116766);
    ref[9] = std::complex<double>(6.342736476721644, -35.00266455224904);
    ref[10] = std::complex<double>(-13.430194051902703, -3.8599450263544313);
    ref[11] = std::complex<double>(-5.257970425183435, 9.979797908837504);
    ref[12] = std::complex<double>(16.247055328357742, 3.102021652590557);
    ref[13] = std::complex<double>(-15.91597417644847, -23.621779576493413);
    ref[14] = std::complex<double>(-23.63840384333899, -5.195086305285825);
    ref[15] = std::complex<double>(7.980540399511438, 11.16381599484859);

    for (std::size_t i = 0; i < outSize; i++) {
        REQUIRE(std::fabs(ref[i] - out[i]) < 1e-5f);
    }
}

TEST_CASE("performCfftBackwardDouble::InvalidTwiddleSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = fft::performCfftBackward<double>(nfft, twiddleFactors.get(), nfft - 1,
                                          in.get(), inSize, out.get(), outSize);
    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftBackwardDouble::InvalidInputSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = fft::performCfftBackward<double>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize - 1, out.get(),
                                          outSize);
    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftBackwardDouble::InvalidOutputSize", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = fft::performCfftBackward<double>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize, out.get(),
                                          outSize - 1);
    REQUIRE(err == fft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftBackwardDouble::NullTwiddle", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = fft::performCfftBackward<double>(nfft, nullptr, nfft, in.get(), inSize,
                                          out.get(), outSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftBackwardDouble::NullInput", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = fft::performCfftBackward<double>(nfft, twiddleFactors.get(), nfft,
                                          nullptr, inSize, out.get(), outSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftBackwardDouble::NullOutput", "[forward]")
{
    fft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = fft::populateCfftTwiddleFactorsBackward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == fft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = fft::performCfftBackward<double>(nfft, twiddleFactors.get(), nfft,
                                           in.get(), inSize, nullptr, outSize);
    REQUIRE(err == fft::FFTSTATUS::NULL_POINTER);
}
