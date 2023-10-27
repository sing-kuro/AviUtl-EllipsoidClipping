#include <lua.hpp>
#include <Eigen/Core>
#include <cmath>
#include <numbers>
#include <iostream>


struct PixelBGRA
{
	uint8_t b, g, r, a;
};

double rad(double degree)
{
	return (degree * std::numbers::pi / 180.0);
}

Eigen::Matrix<double, 3, 3> rotm(double z, double y, double x)
{
	Eigen::Matrix<double, 3, 3> A, B, C;
	A << std::cos(z), -std::sin(z), 0,
		std::sin(z), std::cos(z), 0,
		0, 0, 1;
	B << std::cos(y), 0, std::sin(y),
		0, 1, 0,
		-std::sin(y), 0, std::cos(y);
	C << 1, 0, 0,
		0, std::cos(x), -std::sin(x),
		0, std::sin(x), std::cos(x);

	return A * B * C;
}

int clip(lua_State* L)
{
	int defew = lua_tointeger(L, 1);
	int defeh = lua_tointeger(L, 2);
	int defed = lua_tointeger(L, 3);
	int defpos = lua_tointeger(L, 4);
	double ecx = lua_tonumber(L, 5);
	double ecy = lua_tonumber(L, 6);
	double ew = lua_tonumber(L, 7);
	double eh = lua_tonumber(L, 8);
	double ed = lua_tonumber(L, 9);
	double erz = lua_tonumber(L, 10);
	double ery = lua_tonumber(L, 11);
	double erx = lua_tonumber(L, 12);
	double rz = lua_tonumber(L, 13);
	double ry = lua_tonumber(L, 14);
	double rx = lua_tonumber(L, 15);
	double ext = lua_tonumber(L, 16);
	int flip = lua_toboolean(L, 17);
	PixelBGRA* data = reinterpret_cast<PixelBGRA*>(lua_touserdata(L, 18));
	int wp = lua_tointeger(L, 19);
	int hp = lua_tointeger(L, 20);
	int ox = lua_tointeger(L, 21);
	int oy = lua_tointeger(L, 22);

	if (defew) ew = wp;
	if (defeh) eh = hp;
	if (defed) ed = ew;
	double a = ew / 2;
	double b = eh / 2;
	double c = ed / 2;
	if (defpos)
	{
		ecx = ox + a;
		ecy = oy + b;
	}
	double ecz = std::max(std::max(a, b), c);
	erx = rad(erx);
	ery = rad(ery);
	erz = rad(erz);
	Eigen::Matrix<double, 3, 3> ERot = rotm(erz, ery, erx);
	Eigen::RowVector<double, 3> nv;
	nv << 0, 0, 1;
	nv *= ERot;
	double l = nv(0), m = nv(1), n = nv(2);
	rz = rad(rz);
	ry = rad(ry);
	rx = rad(rx);
	ext = rad(ext);
	auto R = rotm(rz, ry, rx);

#pragma omp parallel for
	for (int i = 0; i < wp * hp; i++)
	{
		int x = i % wp;
		int y = i / wp;
		PixelBGRA pixel = data[i];
		if (pixel.a == 0) continue;

		bool invisible = false;

		Eigen::RowVector<double, 3> dxyz(x - ecx, y - ecy, 0 - ecz);
		dxyz *= ERot;
		double x_fin = dxyz(0), y_fin = dxyz(1), z_fin = dxyz(2);
		double tmp_a = l * l / (a * a) + m * m / (b * b) + n * n / (c * c);
		double tmp_b = 2 * x_fin * l / (a * a) + 2 * y_fin * m / (b * b) + 2 * z_fin * n / (c * c);
		double tmp_c = x_fin * x_fin / (a * a) + y_fin * y_fin / (b * b) + z_fin * z_fin / (c * c) - 1;
		double tmp_d = tmp_b * tmp_b - 4 * tmp_a * tmp_c;

		if (tmp_d >= 0)
		{
			double tmp_t = (-tmp_b - std::sqrt(tmp_d)) / (2 * tmp_a);
			double int_x = x_fin + tmp_t * l;
			double int_y = y_fin + tmp_t * m;
			double int_z = z_fin + tmp_t * n;
			Eigen::RowVector<double, 3> intersec(int_x, int_y, int_z);
			intersec *= R;
			int_x = intersec(0);
			int_y = intersec(1);
			int_z = intersec(2);
			double theta = std::atan2(int_y, int_z);

			if ((0 < theta && theta < ext) || (ext < theta && theta < 0) || (theta + 2 * std::numbers::pi < ext) || ext < theta - 2 * std::numbers::pi)
			{
				invisible = true;
			}
		}
		if (flip) invisible ^= true;
		if (invisible) pixel.a = 0;
		data[i] = pixel;
	}
	return 0;
}

static luaL_Reg functions[] = {
	{"ellipsoid_clipping", clip},
	{nullptr, nullptr}
};

extern "C" {
	__declspec(dllexport) int luaopen_ellipsoid_clipping(lua_State* L) {
		luaL_register(L, "ellipsoid_clipping", functions);
		return 1;
	}
}


