#pragma once

#include <shared_mutex>
#include <iostream>

typedef std::mutex Mutex;
typedef std::shared_mutex Lock;
typedef std::unique_lock<Lock> WriteLock;
typedef std::shared_lock<Lock> ReadLock;