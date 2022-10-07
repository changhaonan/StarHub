// Created by Haonan Chang on 04/05/2022
#pragma once
#include <memory>
#include <string>
#include <sstream>

namespace star
{
    template <typename... Args>
    std::string stringFormat(const std::string &format, Args... args)
    {
        int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
        if (size_s <= 0)
        {
            throw std::runtime_error("Error during formatting.");
        }
        auto size = static_cast<size_t>(size_s);
        auto buf = std::make_unique<char[]>(size);
        std::snprintf(buf.get(), size, format.c_str(), args...);
        return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
    }

    std::string stringAlign2Center(const std::string s, const int w, const char *filling = " ");

}