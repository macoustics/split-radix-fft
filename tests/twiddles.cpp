#include "splitradixfft.hpp"
#include <catch2/catch_test_macros.hpp>
#include <memory>

TEST_CASE("populateCfftTwiddleFactorsForwardFloat::Valid", "[twiddles]")
{
    const std::size_t nfft = 128;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    auto err = splitradixfft::populateCfftTwiddleFactorsForward<float>(nfft, twiddleFactors.get(), nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);
}

TEST_CASE("populateCfftTwiddleFactorsForwardFloat::ErrorOnInvalidSize", "[twiddles]")
{
    const std::size_t nfft = 127;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    auto err = splitradixfft::populateCfftTwiddleFactorsForward<float>(nfft, twiddleFactors.get(), nfft - 1); 
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("populateCfftTwiddleFactorsForwardFloat::ErrorOnNull", "[twiddles]")
{
    const std::size_t nfft = 128;
    std::complex<float>* twiddleFactors = nullptr;
    auto err = splitradixfft::populateCfftTwiddleFactorsForward<float>(nfft, twiddleFactors, nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("populateRfftTwiddleFactorsForwardFloat::Valid", "[twiddles]")
{
    const std::size_t nfft = 128;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    auto err = splitradixfft::populateRfftTwiddleFactorsForward<float>(nfft, twiddleFactors.get(), nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);
}

TEST_CASE("populateRfftTwiddleFactorsForwardFloat::ErrorOnUnevenSize", "[twiddles]")
{
    const std::size_t nfft = 127;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    auto err = splitradixfft::populateRfftTwiddleFactorsForward<float>(nfft, twiddleFactors.get(), nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("populateRfftTwiddleFactorsForwardFloat::ErrorOnInvalidSize", "[twiddles]")
{
    const std::size_t nfft = 128;
    auto twiddleFactors = std::make_unique<std::complex<float>[]>(nfft);
    auto err = splitradixfft::populateRfftTwiddleFactorsForward<float>(nfft, twiddleFactors.get(), nfft - 1); 
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("populateRfftTwiddleFactorsForwardFloat::ErrorOnNull", "[twiddles]")
{
    const std::size_t nfft = 128;
    std::complex<float>* twiddleFactors = nullptr;
    auto err = splitradixfft::populateRfftTwiddleFactorsForward<float>(nfft, twiddleFactors, nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("populateCfftTwiddleFactorsForwardDouble::Valid", "[twiddles]")
{
    const std::size_t nfft = 128;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    auto err = splitradixfft::populateCfftTwiddleFactorsForward<double>(nfft, twiddleFactors.get(), nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);
}

TEST_CASE("populateCfftTwiddleFactorsForwardDouble::ErrorOnInvalidSize", "[twiddles]")
{
    const std::size_t nfft = 127;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    auto err = splitradixfft::populateCfftTwiddleFactorsForward<double>(nfft, twiddleFactors.get(), nfft - 1); 
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("populateCfftTwiddleFactorsForwardDouble::ErrorOnNull", "[twiddles]")
{
    const std::size_t nfft = 128;
    std::complex<double>* twiddleFactors = nullptr;
    auto err = splitradixfft::populateCfftTwiddleFactorsForward<double>(nfft, twiddleFactors, nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}

TEST_CASE("populateRfftTwiddleFactorsForwardDouble::Valid", "[twiddles]")
{
    const std::size_t nfft = 128;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    auto err = splitradixfft::populateRfftTwiddleFactorsForward<double>(nfft, twiddleFactors.get(), nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::OK);
}

TEST_CASE("populateRfftTwiddleFactorsForwardDouble::ErrorOnUnevenSize", "[twiddles]")
{
    const std::size_t nfft = 127;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    auto err = splitradixfft::populateRfftTwiddleFactorsForward<double>(nfft, twiddleFactors.get(), nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("populateRfftTwiddleFactorsForwardDouble::ErrorOnInvalidSize", "[twiddles]")
{
    const std::size_t nfft = 128;
    auto twiddleFactors = std::make_unique<std::complex<double>[]>(nfft);
    auto err = splitradixfft::populateRfftTwiddleFactorsForward<double>(nfft, twiddleFactors.get(), nfft + 1); 
    REQUIRE(err == splitradixfft::FFTSTATUS::INVALID_SIZE);
}

TEST_CASE("populateRfftTwiddleFactorsForwardDouble::ErrorOnNull", "[twiddles]")
{
    const std::size_t nfft = 128;
    std::complex<double>* twiddleFactors = nullptr;
    auto err = splitradixfft::populateRfftTwiddleFactorsForward<double>(nfft, twiddleFactors, nfft); 
    REQUIRE(err == splitradixfft::FFTSTATUS::NULL_POINTER);
}
