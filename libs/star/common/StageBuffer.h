#pragma once
#include <assert.h>
#include <condition_variable>
#include <star/common/Lock.h>

/*
* \brief thread-safe buffer, only allow acquire of buffer in stage order
*/
template<unsigned num_stage>
class StageBuffer {
public:
	using Ptr = std::shared_ptr<StageBuffer>;
	StageBuffer(void** extern_buffer) : stage_idx(0) {
		for (auto i = 0; i < num_stage; ++i) {
			stage_allowed[i] = false;
			stage_initialized[i] = false;
			stage_buffer[i] = extern_buffer[i];
		}
	};
	StageBuffer() : stage_idx(0) {
		for (auto i = 0; i < num_stage; ++i) {
			stage_allowed[i] = false;
			stage_initialized[i] = false;
		}
	};
	~StageBuffer() {};
public:
	void SetBuffer(void* buffer_ptr, const unsigned _stage) {
		stage_buffer[_stage] = buffer_ptr;
	}
	void DeleteBuffer(const unsigned _stage) {
		delete(stage_buffer[_stage]);
	}
public:
	template<typename T>
	T* Buffer(const unsigned _stage, const size_t offset = 0) {
		assert(stage_allowed[_stage]);  // Can only write to allowed buffer
		return (T*)((char*)stage_buffer[_stage] + offset);
	}
	template<typename T>
	const T* BufferReadOnly(const unsigned _stage, const size_t offset = 0) const {
		assert(stage_initialized[_stage] || stage_allowed[_stage]);  // Can only read initialized buffer
		return (const T*)((char*)stage_buffer[_stage] + offset);
	}

public:
	// Thread-control API
	void ColdStart(const unsigned _stage) {
		stage_allowed[_stage] = true;
		stage_initialized[_stage] = true;
	}
	void Wait(const unsigned _stage) {
		if (_stage == 0 && !stage_initialized[_stage]) {
			ColdStart(_stage);
			return;
		}
		Mutex& m = GetMutex(_stage);
		std::condition_variable& cv = GetCV(_stage);
		std::unique_lock<std::mutex> lk(m);
		cv.wait(lk, [&] { return IsAllowed(_stage); });
	}
	void MoveOn() {
		stage_allowed[stage_idx] = false;  // Turn off allowance
		stage_initialized[stage_idx] = true;  // Mark as initialized when entering next stage
		stage_idx = (stage_idx + 1) % num_stage;
		stage_allowed[stage_idx] = true;
		stage_cv[stage_idx].notify_all();
	};  // Notify & Release block bounded with next stage
	
protected:
	Mutex& GetMutex(const unsigned _stage) { return stage_mutex[_stage]; }
	std::condition_variable& GetCV(const unsigned _stage) { return stage_cv[_stage]; }
	bool IsAllowed(const unsigned _stage) { return stage_allowed[_stage]; }
	bool IsInitialized(const unsigned _stage) { return stage_initialized[_stage]; }
protected:
	unsigned stage_idx;
	void* stage_buffer[num_stage];
	Mutex stage_mutex[num_stage];
	std::condition_variable stage_cv[num_stage];
	bool stage_allowed[num_stage];
	bool stage_initialized[num_stage];
};