#pragma once

namespace star
{
	/**
	 * \brief This class is used for counting, whenver a threshold is reached,
	 * it will re-counter from 0.
	 */
	class Counter
	{
	public:
		Counter(const unsigned count_per_round = 10) : m_count(0), m_count_per_round(count_per_round) {}
		void SetCountPerRound(const unsigned count_per_round) { m_count_per_round = count_per_round; }
		int count(const unsigned step = 1)
		{
			m_count += step;
			unsigned round = m_count / m_count_per_round;
			m_count = m_count % m_count_per_round;
			return round;
		}

	private:
		unsigned m_count_per_round;
		unsigned m_count;
	};

}
