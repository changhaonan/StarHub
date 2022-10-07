#pragma once
#include <star/common/Stream.h>
#include <star/common/Serializer.h>

template <typename T>
inline bool star::Stream::SerializeRead(T *output)
{
	return SerializeHandler<T>::Read(this, output);
}

template <typename T>
inline void star::Stream::SerializeWrite(const T &object)
{
	SerializeHandler<T>::Write(this, object);
}