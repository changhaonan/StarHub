#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <istream>
#include <ostream>
#include <streambuf>
#include <star/common/common_types.h>

namespace star
{
	class Stream
	{
	public:
		// Might disable copy/assign
		explicit Stream() = default;
		virtual ~Stream() = default;

		// The read interface
		virtual size_t Read(void *ptr, size_t bytes) = 0;
		template <typename T>
		inline bool SerializeRead(T *output);

		// The write interface
		virtual bool Write(const void *ptr, size_t bytes) = 0;
		template <typename T>
		inline void SerializeWrite(const T &object);
	};
}

#include <star/common/Serializer.h>
#include <star/common/Stream.hpp>