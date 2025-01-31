#include "splitradixfft.hpp"
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

TEST_CASE("performCfftForwardFloat::Valid", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    in[0] = std::complex<float>(0.3846678550174167, 0.9328994221406891);
    in[1] = std::complex<float>(-0.7183560201867548, -1.9923064312047984);
    in[2] = std::complex<float>(-1.7971202366185075, -1.4839469697949352);
    in[3] = std::complex<float>(-0.4871520113970549, 0.43234613045395937);
    in[4] = std::complex<float>(-0.9056041125877594, -0.24590133007204354);
    in[5] = std::complex<float>(1.4375823908503307, 1.4877463942012574);
    in[6] = std::complex<float>(0.4441534315934983, 0.5988912738664438);
    in[7] = std::complex<float>(1.3564267648119048, -0.5284815927550913);
    in[8] = std::complex<float>(0.9237893776133843, 0.5975417394479786);
    in[9] = std::complex<float>(0.39642102979510285, -2.1876665345155644);
    in[10] = std::complex<float>(-0.8393871282439191, -0.24124656414715218);
    in[11] = std::complex<float>(-0.32862315157396454, 0.6237373693023441);
    in[12] = std::complex<float>(1.015440958022359, 0.1938763532869097);
    in[13] = std::complex<float>(-0.9947483860280294, -1.4763612235308383);
    in[14] = std::complex<float>(-1.4774002402086863, -0.3246928940803639);
    in[15] = std::complex<float>(0.49878377496946474, 0.697738499678037);

    err = splitradixfft::performCfftForward<float>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize, out.get(), outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t refSize = nfft;
    auto ref = std::make_unique<std::complex<float>[]>(outSize);

    ref[0] = std::complex<float>(-1.0911257041712146, -2.9158263577231676);
    ref[1] = std::complex<float>(-3.8871063587373555, -1.8534222089417363);
    ref[2] = std::complex<float>(-1.7895294293147468, 2.0243532818942183);
    ref[3] = std::complex<float>(1.336336765713034, 3.8316062823627943);
    ref[4] = std::complex<float>(-0.30587995018617775, 3.8479477013392422);
    ref[5] = std::complex<float>(-2.369579527779574, 2.1181125417824473);
    ref[6] = std::complex<float>(-0.48654312248145626, 5.999469885704463);
    ref[7] = std::complex<float>(3.07126264502644, -4.544087664490265);
    ref[8] = std::complex<float>(-3.4117944866532133, 2.970668419018221);
    ref[9] = std::complex<float>(-2.5939169789902357, 1.9395905791169208);
    ref[10] = std::complex<float>(0.18798637625081538, 4.3471001073478615);
    ref[11] = std::complex<float>(2.085603458970295, -5.302440146753907);
    ref[12] = std::complex<float>(10.481976453272209, 2.0108749765798404);
    ref[13] = std::complex<float>(4.9350060416874815, 6.821330101253683);
    ref[14] = std::complex<float>(6.882567724330194, -6.0410587214513365);
    ref[15] = std::complex<float>(-6.890578226657826, -0.3278280227882546);

    for (std::size_t i = 0; i < outSize; i++) {
        REQUIRE(std::fabs(ref[i] - out[i]) < 1e-5f);
    }

}

TEST_CASE("performCfftForwardFloat::InvalidTwiddleSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = splitradixfft::performCfftForward<float>(nfft, twiddleFactors.get(), nfft - 1,
                                         in.get(), inSize, out.get(), outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftForwardFloat::InvalidInputSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = splitradixfft::performCfftForward<float>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize - 1, out.get(),
                                         outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftForwardFloat::InvalidOutputSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = splitradixfft::performCfftForward<float>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize, out.get(),
                                         outSize - 1);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftForwardFloat::NullTwiddle", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = splitradixfft::performCfftForward<float>(nfft, nullptr, nfft, in.get(), inSize,
                                         out.get(), outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftForwardFloat::NullInput", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = splitradixfft::performCfftForward<float>(nfft, twiddleFactors.get(), nfft,
                                         nullptr, inSize, out.get(), outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftForwardFloat::NullOutput", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<float>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<float>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<float>[]>(outSize);

    err = splitradixfft::performCfftForward<float>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize, nullptr, outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftForwardDouble::Valid", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    in[0] = std::complex<double>(0.3846678550174167, 0.9328994221406891);
    in[1] = std::complex<double>(-0.7183560201867548, -1.9923064312047984);
    in[2] = std::complex<double>(-1.7971202366185075, -1.4839469697949352);
    in[3] = std::complex<double>(-0.4871520113970549, 0.43234613045395937);
    in[4] = std::complex<double>(-0.9056041125877594, -0.24590133007204354);
    in[5] = std::complex<double>(1.4375823908503307, 1.4877463942012574);
    in[6] = std::complex<double>(0.4441534315934983, 0.5988912738664438);
    in[7] = std::complex<double>(1.3564267648119048, -0.5284815927550913);
    in[8] = std::complex<double>(0.9237893776133843, 0.5975417394479786);
    in[9] = std::complex<double>(0.39642102979510285, -2.1876665345155644);
    in[10] = std::complex<double>(-0.8393871282439191, -0.24124656414715218);
    in[11] = std::complex<double>(-0.32862315157396454, 0.6237373693023441);
    in[12] = std::complex<double>(1.015440958022359, 0.1938763532869097);
    in[13] = std::complex<double>(-0.9947483860280294, -1.4763612235308383);
    in[14] = std::complex<double>(-1.4774002402086863, -0.3246928940803639);
    in[15] = std::complex<double>(0.49878377496946474, 0.697738499678037);

    err = splitradixfft::performCfftForward<double>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize, out.get(), outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t refSize = nfft;
    auto ref = std::make_unique<std::complex<double>[]>(outSize);

    ref[0] = std::complex<double>(-1.0911257041712146, -2.9158263577231676);
    ref[1] = std::complex<double>(-3.8871063587373555, -1.8534222089417363);
    ref[2] = std::complex<double>(-1.7895294293147468, 2.0243532818942183);
    ref[3] = std::complex<double>(1.336336765713034, 3.8316062823627943);
    ref[4] = std::complex<double>(-0.30587995018617775, 3.8479477013392422);
    ref[5] = std::complex<double>(-2.369579527779574, 2.1181125417824473);
    ref[6] = std::complex<double>(-0.48654312248145626, 5.999469885704463);
    ref[7] = std::complex<double>(3.07126264502644, -4.544087664490265);
    ref[8] = std::complex<double>(-3.4117944866532133, 2.970668419018221);
    ref[9] = std::complex<double>(-2.5939169789902357, 1.9395905791169208);
    ref[10] = std::complex<double>(0.18798637625081538, 4.3471001073478615);
    ref[11] = std::complex<double>(2.085603458970295, -5.302440146753907);
    ref[12] = std::complex<double>(10.481976453272209, 2.0108749765798404);
    ref[13] = std::complex<double>(4.9350060416874815, 6.821330101253683);
    ref[14] = std::complex<double>(6.882567724330194, -6.0410587214513365);
    ref[15] = std::complex<double>(-6.890578226657826, -0.3278280227882546);

    for (std::size_t i = 0; i < outSize; i++) {
        REQUIRE(std::fabs(ref[i] - out[i]) < 1e-5f);
    }
}

TEST_CASE("performCfftForwardDouble::InvalidTwiddleSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = splitradixfft::performCfftForward<double>(nfft, twiddleFactors.get(), nfft - 1,
                                         in.get(), inSize, out.get(), outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftForwardDouble::InvalidInputSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = splitradixfft::performCfftForward<double>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize - 1, out.get(),
                                         outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftForwardDouble::InvalidOutputSize", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = splitradixfft::performCfftForward<double>(nfft, twiddleFactors.get(), nfft,
                                         in.get(), inSize, out.get(),
                                         outSize - 1);
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("performCfftForwardDouble::NullTwiddle", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = splitradixfft::performCfftForward<double>(nfft, nullptr, nfft, in.get(), inSize,
                                         out.get(), outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftForwardDouble::NullInput", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = splitradixfft::performCfftForward<double>(nfft, twiddleFactors.get(), nfft,
                                         nullptr, inSize, out.get(), outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("performCfftForwardDouble::NullOutput", "[forward]")
{
    splitradixfft::FFTSTATUS err;
    const std::size_t nfft = 16;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    err = splitradixfft::populateCfftTwiddleFactorsForward<double>(
        nfft, twiddleFactors.get(), nfft);
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);

    const std::size_t inSize = nfft;
    auto in = std::make_unique<std::complex<double>[]>(inSize);

    const std::size_t outSize = nfft;
    auto out = std::make_unique<std::complex<double>[]>(outSize);

    err = splitradixfft::performCfftForward<double>(nfft, twiddleFactors.get(), nfft,
                                          in.get(), inSize, nullptr, outSize);
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}
