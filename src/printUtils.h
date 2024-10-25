#ifndef __PRINT_UTILS_H__
#define __PRINT_UTILS_H__

// debug
#define dprintf(format, ...) do {                                                       \
        printf(("\033[96m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__);    \
        fflush(stdout);                                                                 \
    } while (false)
// info
#define iprintf(format, ...) do {                                                       \
        printf(("\033[92m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__);    \
        fflush(stdout);                                                                 \
    } while (false)
// notice
#define nprintf(format, ...) do {                                                       \
        printf(("\033[94m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__);    \
        fflush(stdout);                                                                 \
    } while (false)
// warning
#define wprintf(format, ...) do {                                                       \
        printf(("\033[93m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__);    \
        fflush(stdout);                                                                 \
    } while (false)
// error
#define eprintf(format, ...) do {                                                       \
        printf(("\033[91m" + std::string(format) + "\033[0m").c_str(), __VA_ARGS__);    \
        fflush(stdout);                                                                 \
    } while (false)

#endif
