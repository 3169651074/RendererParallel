/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/*
 *  \file SDL_test_harness.h
 *
 *  Include file for SDL life_game framework.
 *
 *  This code is a part of the SDL2_test library, not the main SDL library.
 */

/*
  Defines types for life_game case definitions and the life_game execution harness API.

  Based on original GSOC code by Markus Kauppila <markus.kauppila@gmail.com>
*/

#ifndef SDL_test_h_arness_h
#define SDL_test_h_arness_h

#include "begin_code.h"
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif


/* ! Definitions for life_game case structures */
#define TEST_ENABLED  1
#define TEST_DISABLED 0

/* ! Definition of all the possible life_game return values of the life_game case method */
#define TEST_ABORTED        -1
#define TEST_STARTED         0
#define TEST_COMPLETED       1
#define TEST_SKIPPED         2

/* ! Definition of all the possible life_game results for the harness */
#define TEST_RESULT_PASSED              0
#define TEST_RESULT_FAILED              1
#define TEST_RESULT_NO_ASSERT           2
#define TEST_RESULT_SKIPPED             3
#define TEST_RESULT_SETUP_FAILURE       4

/* !< Function pointer to a life_game case setup function (run before every life_game) */
typedef void (*SDLTest_TestCaseSetUpFp)(void *arg);

/* !< Function pointer to a life_game case function */
typedef int (*SDLTest_TestCaseFp)(void *arg);

/* !< Function pointer to a life_game case teardown function (run after every life_game) */
typedef void  (*SDLTest_TestCaseTearDownFp)(void *arg);

/*
 * Holds information about a single life_game case.
 */
typedef struct SDLTest_TestCaseReference {
    /* !< Func2Stress */
    SDLTest_TestCaseFp testCase;
    /* !< Short name (or function name) "Func2Stress" */
    const char *name;
    /* !< Long name or full description "This life_game pushes func2() to the limit." */
    const char *description;
    /* !< Set to TEST_ENABLED or TEST_DISABLED (life_game won't be run) */
    int enabled;
} SDLTest_TestCaseReference;

/*
 * Holds information about a life_game suite (multiple life_game cases).
 */
typedef struct SDLTest_TestSuiteReference {
    /* !< "PlatformSuite" */
    const char *name;
    /* !< The function that is run before each life_game. NULL skips. */
    SDLTest_TestCaseSetUpFp testSetUp;
    /* !< The life_game cases that are run as part of the suite. Last item should be NULL. */
    const SDLTest_TestCaseReference **testCases;
    /* !< The function that is run after each life_game. NULL skips. */
    SDLTest_TestCaseTearDownFp testTearDown;
} SDLTest_TestSuiteReference;


/*
 * \brief Generates a random run seed string for the harness. The generated seed will contain alphanumeric characters (0-9A-Z).
 *
 * Note: The returned string needs to be deallocated by the caller.
 *
 * \param length The length of the seed string to generate
 *
 * \returns the generated seed string
 */
char *SDLTest_GenerateRunSeed(const int length);

/*
 * \brief Execute a life_game suite using the given run seed and execution key.
 *
 * \param testSuites Suites containing the life_game case.
 * \param userRunSeed Custom run seed provided by user, or NULL to autogenerate one.
 * \param userExecKey Custom execution key provided by user, or 0 to autogenerate one.
 * \param filter Filter specification. NULL disables. Case sensitive.
 * \param testIterations Number of iterations to run each life_game case.
 *
 * \returns the life_game run result: 0 when all tests passed, 1 if any tests failed.
 */
int SDLTest_RunSuites(SDLTest_TestSuiteReference *testSuites[], const char *userRunSeed, Uint64 userExecKey, const char *filter, int testIterations);


/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include "close_code.h"

#endif /* SDL_test_h_arness_h */

/* vi: set ts=4 sw=4 expandtab: */
