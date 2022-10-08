#pragma once

namespace star
{
    /**
     * \brief GeometryMap is the base class for 2D geometry
     */
    class GeometryMap
    {
    public:
        GeometryMap(const unsigned width, const unsigned height) : m_width(width), m_height(height){};
        virtual bool IsEmpty();

    protected:
        unsigned m_width;
        unsigned m_height;
    };

}