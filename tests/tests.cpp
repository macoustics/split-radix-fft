#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Example tests", "[example]" ) {
    REQUIRE( true == true );
}
