#include <star/common/string_utils.hpp>

std::string star::stringAlign2Center(const std::string s, const int w, const char *filling)
{
    std::stringstream ss, spaces;
    int pad = w - s.size(); // count excess room to pad
    for (int i = 0; i < pad / 2; ++i)
        spaces << filling;
    ss << spaces.str() << s << spaces.str(); // format with padding
    if (pad > 0 && pad % 2 != 0)             // if pad odd #, add 1 more space
        ss << filling;
    return ss.str();
    return s;
}