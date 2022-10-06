#pragma once

#define STAR_NO_COPY_ASSIGN(TypeName)    \
    TypeName(const TypeName&) = delete;        \
    TypeName& operator=(const TypeName&) = delete

#define STAR_NO_COPY_ASSIGN_MOVE(TypeName)   \
    TypeName(const TypeName&) = delete;            \
    TypeName& operator=(const TypeName&) = delete; \
    TypeName(TypeName&&) = delete;                 \
    TypeName& operator=(TypeName&&) = delete

#define STAR_DEFAULT_MOVE(TypeName) \
	TypeName(TypeName&&) noexcept = default;       \
	TypeName& operator=(TypeName&&) noexcept = default

#define STAR_DEFAULT_CONSTRUCT_DESTRUCT(TypeName) \
    TypeName() = default;                               \
    ~TypeName() = default

