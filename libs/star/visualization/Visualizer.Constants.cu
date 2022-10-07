#include <star/visualization/Visualizer.h>

const std::map<int, uchar3> star::visualize::default_semantic_color_dict = {
	{0, make_uchar3(128, 128, 128)},
	{1, make_uchar3(255, 0, 0)},
	{2, make_uchar3(255, 140, 0)},
	{3, make_uchar3(255, 215, 0)},
	{4, make_uchar3(128, 128, 0)},
	{5, make_uchar3(255, 255, 0)},
	{6, make_uchar3(85, 107, 47)},
	{7, make_uchar3(124, 252, 0)},
	{8, make_uchar3(0, 255, 0)},
	{9, make_uchar3(47, 79, 79)},
	{10, make_uchar3(0, 128, 128)},
	{11, make_uchar3(0, 255, 255)},
	{12, make_uchar3(100, 149, 237)},
	{13, make_uchar3(0, 191, 255)},
	{14, make_uchar3(0, 0, 255)},
	{15, make_uchar3(138, 43, 226)},
	{16, make_uchar3(255, 0, 255)},
	{17, make_uchar3(139, 69, 19)},
	{18, make_uchar3(210, 105, 30)}
};