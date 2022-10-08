#pragma once

namespace star
{
    /**
     * \brief GeometryMap is the base class for 2D geometry
     */
    class GeometryMap
    {
    public:
        GeometryMap(const unsigned width, const unsigned height) : m_width(width), m_height(height) { m_num_pixel = width * height; };
        virtual bool IsEmpty();
        unsigned NumPixel() const { return m_num_pixel; }

    protected:
        unsigned m_width;
        unsigned m_height;
        unsigned m_num_pixel;
    };

}