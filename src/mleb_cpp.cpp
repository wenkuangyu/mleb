#include <RcppArmadillo.h>
#include <R_ext/Applic.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// ============================================================================
// SECTION 1: Gauss-Legendre Quadrature & Plug-in Selector
// ============================================================================

static const double LEGENDRE_X[500] = {
  0.999988457,  0.999939180,  0.999850529,  0.999722488,  0.999555059,  0.999348249,
  0.999102065,  0.998816517,  0.998491616,  0.998127375,  0.997723809,  0.997280933,
  0.996798765,  0.996277323,  0.995716628,  0.995116703,  0.994477571,  0.993799257,
  0.993081788,  0.992325192,  0.991529499,  0.990694740,  0.989820948,  0.988908158,
  0.987956405,  0.986965728,  0.985936165,  0.984867756,  0.983760544,  0.982614572,
  0.981429886,  0.980206532,  0.978944559,  0.977644016,  0.976304954,  0.974927426,
  0.973511486,  0.972057191,  0.970564597,  0.969033764,  0.967464751,  0.965857620,
  0.964212436,  0.962529261,  0.960808164,  0.959049212,  0.957252473,  0.955418019,
  0.953545922,  0.951636256,  0.949689096,  0.947704519,  0.945682603,  0.943623428,
  0.941527074,  0.939393624,  0.937223163,  0.935015776,  0.932771550,  0.930490573,
  0.928172935,  0.925818728,  0.923428044,  0.921000978,  0.918537624,  0.916038081,
  0.913502446,  0.910930820,  0.908323303,  0.905680000,  0.903001012,  0.900286448,
  0.897536412,  0.894751014,  0.891930363,  0.889074571,  0.886183749,  0.883258013,
  0.880297476,  0.877302256,  0.874272472,  0.871208241,  0.868109685,  0.864976926,
  0.861810087,  0.858609294,  0.855374672,  0.852106348,  0.848804452,  0.845469114,
  0.842100465,  0.838698637,  0.835263765,  0.831795984,  0.828295431,  0.824762243,
  0.821196561,  0.817598523,  0.813968273,  0.810305953,  0.806611707,  0.802885681,
  0.799128022,  0.795338877,  0.791518397,  0.787666731,  0.783784032,  0.779870452,
  0.775926145,  0.771951268,  0.767945976,  0.763910427,  0.759844781,  0.755749197,
  0.751623837,  0.747468864,  0.743284441,  0.739070732,  0.734827905,  0.730556126,
  0.726255564,  0.721926387,  0.717568767,  0.713182875,  0.708768884,  0.704326968,
  0.699857302,  0.695360061,  0.690835424,  0.686283569,  0.681704674,  0.677098921,
  0.672466490,  0.667807564,  0.663122327,  0.658410964,  0.653673659,  0.648910600,
  0.644121975,  0.639307971,  0.634468779,  0.629604589,  0.624715593,  0.619801984,
  0.614863955,  0.609901700,  0.604915416,  0.599905298,  0.594871545,  0.589814353,
  0.584733924,  0.579630456,  0.574504151,  0.569355211,  0.564183839,  0.558990238,
  0.553774613,  0.548537170,  0.543278115,  0.537997655,  0.532695998,  0.527373353,
  0.522029930,  0.516665940,  0.511281593,  0.505877101,  0.500452679,  0.495008539,
  0.489544895,  0.484061964,  0.478559962,  0.473039104,  0.467499609,  0.461941694,
  0.456365579,  0.450771484,  0.445159629,  0.439530234,  0.433883523,  0.428219716,
  0.422539038,  0.416841712,  0.411127963,  0.405398015,  0.399652095,  0.393890430,
  0.388113244,  0.382320768,  0.376513228,  0.370690854,  0.364853875,  0.359002521,
  0.353137022,  0.347257610,  0.341364516,  0.335457973,  0.329538212,  0.323605468,
  0.317659975,  0.311701965,  0.305731675,  0.299749339,  0.293755193,  0.287749473,
  0.281732416,  0.275704259,  0.269665240,  0.263615595,  0.257555565,  0.251485387,
  0.245405300,  0.239315545,  0.233216361,  0.227107988,  0.220990667,  0.214864639,
  0.208730146,  0.202587429,  0.196436730,  0.190278292,  0.184112356,  0.177939167,
  0.171758967,  0.165572000,  0.159378510,  0.153178740,  0.146972935,  0.140761339,
  0.134544197,  0.128321754,  0.122094256,  0.115861947,  0.109625073,  0.103383880,
  0.097138614,  0.090889520,  0.084636846,  0.078380837,  0.072121740,  0.065859801,
  0.059595267,  0.053328385,  0.047059403,  0.040788566,  0.034516122,  0.028242318,
  0.021967401,  0.015691619,  0.009415219,  0.003138447, -0.003138447, -0.009415219,
  -0.015691619, -0.021967401, -0.028242318, -0.034516122, -0.040788566, -0.047059403,
  -0.053328385, -0.059595267, -0.065859801, -0.072121740, -0.078380837, -0.084636846,
  -0.090889520, -0.097138614, -0.103383880, -0.109625073, -0.115861947, -0.122094256,
  -0.128321754, -0.134544197, -0.140761339, -0.146972935, -0.153178740, -0.159378510,
  -0.165572000, -0.171758967, -0.177939167, -0.184112356, -0.190278292, -0.196436730,
  -0.202587429, -0.208730146, -0.214864639, -0.220990667, -0.227107988, -0.233216361,
  -0.239315545, -0.245405300, -0.251485387, -0.257555565, -0.263615595, -0.269665240,
  -0.275704259, -0.281732416, -0.287749473, -0.293755193, -0.299749339, -0.305731675,
  -0.311701965, -0.317659975, -0.323605468, -0.329538212, -0.335457973, -0.341364516,
  -0.347257610, -0.353137022, -0.359002521, -0.364853875, -0.370690854, -0.376513228,
  -0.382320768, -0.388113244, -0.393890430, -0.399652095, -0.405398015, -0.411127963,
  -0.416841712, -0.422539038, -0.428219716, -0.433883523, -0.439530234, -0.445159629,
  -0.450771484, -0.456365579, -0.461941694, -0.467499609, -0.473039104, -0.478559962,
  -0.484061964, -0.489544895, -0.495008539, -0.500452679, -0.505877101, -0.511281593,
  -0.516665940, -0.522029930, -0.527373353, -0.532695998, -0.537997655, -0.543278115,
  -0.548537170, -0.553774613, -0.558990238, -0.564183839, -0.569355211, -0.574504151,
  -0.579630456, -0.584733924, -0.589814353, -0.594871545, -0.599905298, -0.604915416,
  -0.609901700, -0.614863955, -0.619801984, -0.624715593, -0.629604589, -0.634468779,
  -0.639307971, -0.644121975, -0.648910600, -0.653673659, -0.658410964, -0.663122327,
  -0.667807564, -0.672466490, -0.677098921, -0.681704674, -0.686283569, -0.690835424,
  -0.695360061, -0.699857302, -0.704326968, -0.708768884, -0.713182875, -0.717568767,
  -0.721926387, -0.726255564, -0.730556126, -0.734827905, -0.739070732, -0.743284441,
  -0.747468864, -0.751623837, -0.755749197, -0.759844781, -0.763910427, -0.767945976,
  -0.771951268, -0.775926145, -0.779870452, -0.783784032, -0.787666731, -0.791518397,
  -0.795338877, -0.799128022, -0.802885681, -0.806611707, -0.810305953, -0.813968273,
  -0.817598523, -0.821196561, -0.824762243, -0.828295431, -0.831795984, -0.835263765,
  -0.838698637, -0.842100465, -0.845469114, -0.848804452, -0.852106348, -0.855374672,
  -0.858609294, -0.861810087, -0.864976926, -0.868109685, -0.871208241, -0.874272472,
  -0.877302256, -0.880297476, -0.883258013, -0.886183749, -0.889074571, -0.891930363,
  -0.894751014, -0.897536412, -0.900286448, -0.903001012, -0.905680000, -0.908323303,
  -0.910930820, -0.913502446, -0.916038081, -0.918537624, -0.921000978, -0.923428044,
  -0.925818728, -0.928172935, -0.930490573, -0.932771550, -0.935015776, -0.937223163,
  -0.939393624, -0.941527074, -0.943623428, -0.945682603, -0.947704519, -0.949689096,
  -0.951636256, -0.953545922, -0.955418019, -0.957252473, -0.959049212, -0.960808164,
  -0.962529261, -0.964212436, -0.965857620, -0.967464751, -0.969033764, -0.970564597,
  -0.972057191, -0.973511486, -0.974927426, -0.976304954, -0.977644016, -0.978944559,
  -0.980206532, -0.981429886, -0.982614572, -0.983760544, -0.984867756, -0.985936165,
  -0.986965728, -0.987956405, -0.988908158, -0.989820948, -0.990694740, -0.991529499,
  -0.992325192, -0.993081788, -0.993799257, -0.994477571, -0.995116703, -0.995716628,
  -0.996277323, -0.996798765, -0.997280933, -0.997723809, -0.998127375, -0.998491616,
  -0.998816517, -0.999102065, -0.999348249, -0.999555059, -0.999722488, -0.999850529,
  -0.99939180, -0.999988457
};

static const double LEGENDRE_W[500] = {
  2.962364e-05, 6.895707e-05, 1.083460e-04, 1.477358e-04, 1.871207e-04, 2.264986e-04,
  2.658677e-04, 3.052263e-04, 3.445729e-04, 3.839060e-04, 4.232240e-04, 4.625253e-04,
  5.018083e-04, 5.410716e-04, 5.803136e-04, 6.195328e-04, 6.587275e-04, 6.978962e-04,
  7.370375e-04, 7.761497e-04, 8.152314e-04, 8.542809e-04, 8.932968e-04, 9.322774e-04,
  9.712214e-04, 1.010127e-03, 1.048993e-03, 1.087817e-03, 1.126599e-03, 1.165336e-03,
  1.204028e-03, 1.242672e-03, 1.281267e-03, 1.319811e-03, 1.358304e-03, 1.396743e-03,
  1.435127e-03, 1.473454e-03, 1.511724e-03, 1.549933e-03, 1.588082e-03, 1.626168e-03,
  1.664190e-03, 1.702147e-03, 1.740036e-03, 1.777857e-03, 1.815608e-03, 1.853287e-03,
  1.890894e-03, 1.928426e-03, 1.965881e-03, 2.003260e-03, 2.040559e-03, 2.077778e-03,
  2.114916e-03, 2.151969e-03, 2.188938e-03, 2.225821e-03, 2.262616e-03, 2.299322e-03,
  2.335938e-03, 2.372461e-03, 2.408891e-03, 2.445226e-03, 2.481465e-03, 2.517606e-03,
  2.553647e-03, 2.589588e-03, 2.625427e-03, 2.661163e-03, 2.696794e-03, 2.732318e-03,
  2.767735e-03, 2.803043e-03, 2.838240e-03, 2.873326e-03, 2.908298e-03, 2.943156e-03,
  2.977898e-03, 3.012522e-03, 3.047028e-03, 3.081414e-03, 3.115678e-03, 3.149819e-03,
  3.183837e-03, 3.217729e-03, 3.251494e-03, 3.285131e-03, 3.318639e-03, 3.352016e-03,
  3.385261e-03, 3.418372e-03, 3.451349e-03, 3.484190e-03, 3.516893e-03, 3.549458e-03,
  3.581884e-03, 3.614168e-03, 3.646309e-03, 3.678307e-03, 3.710160e-03, 3.741867e-03,
  3.773427e-03, 3.804837e-03, 3.836098e-03, 3.867208e-03, 3.898165e-03, 3.928969e-03,
  3.959618e-03, 3.990111e-03, 4.020447e-03, 4.050624e-03, 4.080642e-03, 4.110499e-03,
  4.140194e-03, 4.169726e-03, 4.199093e-03, 4.228296e-03, 4.257331e-03, 4.286199e-03,
  4.314898e-03, 4.343427e-03, 4.371785e-03, 4.399970e-03, 4.427983e-03, 4.455820e-03,
  4.483483e-03, 4.510968e-03, 4.538276e-03, 4.565405e-03, 4.592354e-03, 4.619122e-03,
  4.645709e-03, 4.672112e-03, 4.698331e-03, 4.724365e-03, 4.750213e-03, 4.775874e-03,
  4.801346e-03, 4.826630e-03, 4.851723e-03, 4.876625e-03, 4.901335e-03, 4.925852e-03,
  4.950174e-03, 4.974302e-03, 4.998234e-03, 5.021969e-03, 5.045506e-03, 5.068844e-03,
  5.091982e-03, 5.114920e-03, 5.137656e-03, 5.160190e-03, 5.182521e-03, 5.204647e-03,
  5.226568e-03, 5.248284e-03, 5.269792e-03, 5.291093e-03, 5.312186e-03, 5.333069e-03,
  5.353742e-03, 5.374204e-03, 5.394455e-03, 5.414492e-03, 5.434317e-03, 5.453927e-03,
  5.473323e-03, 5.492503e-03, 5.511466e-03, 5.530213e-03, 5.548741e-03, 5.567051e-03,
  5.585142e-03, 5.603012e-03, 5.620662e-03, 5.638090e-03, 5.655296e-03, 5.672280e-03,
  5.689039e-03, 5.705575e-03, 5.721886e-03, 5.737971e-03, 5.753831e-03, 5.769463e-03,
  5.784869e-03, 5.800046e-03, 5.814995e-03, 5.829715e-03, 5.844205e-03, 5.858465e-03,
  5.872494e-03, 5.886292e-03, 5.899857e-03, 5.913191e-03, 5.926291e-03, 5.939158e-03,
  5.951791e-03, 5.964189e-03, 5.976352e-03, 5.988280e-03, 5.999972e-03, 6.011428e-03,
  6.022646e-03, 6.033628e-03, 6.044372e-03, 6.054877e-03, 6.065144e-03, 6.075172e-03,
  6.084961e-03, 6.094510e-03, 6.103819e-03, 6.112887e-03, 6.121714e-03, 6.130301e-03,
  6.138645e-03, 6.146748e-03, 6.154609e-03, 6.162227e-03, 6.169603e-03, 6.176735e-03,
  6.183624e-03, 6.190269e-03, 6.196671e-03, 6.202828e-03, 6.208741e-03, 6.214410e-03,
  6.219833e-03, 6.225011e-03, 6.229945e-03, 6.234632e-03, 6.239074e-03, 6.243271e-03,
  6.247221e-03, 6.250925e-03, 6.254383e-03, 6.257594e-03, 6.260559e-03, 6.263277e-03,
  6.265749e-03, 6.267973e-03, 6.269951e-03, 6.271682e-03, 6.273165e-03, 6.274401e-03,
  6.275391e-03, 6.276132e-03, 6.276627e-03, 6.276874e-03, 6.276874e-03, 6.276627e-03,
  6.276132e-03, 6.275391e-03, 6.274401e-03, 6.273165e-03, 6.271682e-03, 6.269951e-03,
  6.267973e-03, 6.265749e-03, 6.263277e-03, 6.260559e-03, 6.257594e-03, 6.254383e-03,
  6.250925e-03, 6.247221e-03, 6.243271e-03, 6.239074e-03, 6.234632e-03, 6.229945e-03,
  6.225011e-03, 6.219833e-03, 6.214410e-03, 6.208741e-03, 6.202828e-03, 6.196671e-03,
  6.190269e-03, 6.183624e-03, 6.176735e-03, 6.169603e-03, 6.162227e-03, 6.154609e-03,
  6.146748e-03, 6.138645e-03, 6.130301e-03, 6.121714e-03, 6.112887e-03, 6.103819e-03,
  6.094510e-03, 6.084961e-03, 6.075172e-03, 6.065144e-03, 6.054877e-03, 6.044372e-03,
  6.033628e-03, 6.022646e-03, 6.011428e-03, 5.999972e-03, 5.988280e-03, 5.976352e-03,
  5.964189e-03, 5.951791e-03, 5.939158e-03, 5.926291e-03, 5.913191e-03, 5.899857e-03,
  5.886292e-03, 5.872494e-03, 5.858465e-03, 5.844205e-03, 5.829715e-03, 5.814995e-03,
  5.800046e-03, 5.784869e-03, 5.769463e-03, 5.753831e-03, 5.737971e-03, 5.721886e-03,
  5.705575e-03, 5.689039e-03, 5.672280e-03, 5.655296e-03, 5.638090e-03, 5.620662e-03,
  5.603012e-03, 5.585142e-03, 5.567051e-03, 5.548741e-03, 5.530213e-03, 5.511466e-03,
  5.492503e-03, 5.473323e-03, 5.453927e-03, 5.434317e-03, 5.414492e-03, 5.394455e-03,
  5.374204e-03, 5.353742e-03, 5.333069e-03, 5.312186e-03, 5.291093e-03, 5.269792e-03,
  5.248284e-03, 5.226568e-03, 5.204647e-03, 5.182521e-03, 5.160190e-03, 5.137656e-03,
  5.114920e-03, 5.091982e-03, 5.068844e-03, 5.045506e-03, 5.021969e-03, 4.998234e-03,
  4.974302e-03, 4.950174e-03, 4.925852e-03, 4.901335e-03, 4.876625e-03, 4.851723e-03,
  4.826630e-03, 4.801346e-03, 4.775874e-03, 4.750213e-03, 4.724365e-03, 4.698331e-03,
  4.672112e-03, 4.645709e-03, 4.619122e-03, 4.592354e-03, 4.565405e-03, 4.538276e-03,
  4.510968e-03, 4.483483e-03, 4.455820e-03, 4.427983e-03, 4.399970e-03, 4.371785e-03,
  4.343427e-03, 4.314898e-03, 4.286199e-03, 4.257331e-03, 4.228296e-03, 4.199093e-03,
  4.169726e-03, 4.140194e-03, 4.110499e-03, 4.080642e-03, 4.050624e-03, 4.020447e-03,
  3.990111e-03, 3.959618e-03, 3.928969e-03, 3.898165e-03, 3.867208e-03, 3.836098e-03,
  3.804837e-03, 3.773427e-03, 3.741867e-03, 3.710160e-03, 3.678307e-03, 3.646309e-03,
  3.614168e-03, 3.581884e-03, 3.549458e-03, 3.516893e-03, 3.484190e-03, 3.451349e-03,
  3.418372e-03, 3.385261e-03, 3.352016e-03, 3.318639e-03, 3.285131e-03, 3.251494e-03,
  3.217729e-03, 3.183837e-03, 3.149819e-03, 3.115678e-03, 3.081414e-03, 3.047028e-03,
  3.012522e-03, 2.977898e-03, 2.943156e-03, 2.908298e-03, 2.873326e-03, 2.838240e-03,
  2.803043e-03, 2.767735e-03, 2.732318e-03, 2.696794e-03, 2.661163e-03, 2.625427e-03,
  2.589588e-03, 2.553647e-03, 2.517606e-03, 2.481465e-03, 2.445226e-03, 2.408891e-03,
  2.372461e-03, 2.335938e-03, 2.299322e-03, 2.262616e-03, 2.225821e-03, 2.188938e-03,
  2.151969e-03, 2.114916e-03, 2.077778e-03, 2.040559e-03, 2.003260e-03, 1.965881e-03,
  1.928426e-03, 1.890894e-03, 1.853287e-03, 1.815608e-03, 1.777857e-03, 1.740036e-03,
  1.702147e-03, 1.664190e-03, 1.626168e-03, 1.588082e-03, 1.549933e-03, 1.511724e-03,
  1.473454e-03, 1.435127e-03, 1.396743e-03, 1.358304e-03, 1.319811e-03, 1.281267e-03,
  1.242672e-03, 1.204028e-03, 1.165336e-03, 1.126599e-03, 1.087817e-03, 1.048993e-03,
  1.010127e-03, 9.712214e-04, 9.322774e-04, 8.932968e-04, 8.542809e-04, 8.152314e-04,
  7.761497e-04, 7.370375e-04, 6.978962e-04, 6.587275e-04, 6.195328e-04, 5.803136e-04,
  5.410716e-04, 5.018083e-04, 4.625253e-04, 4.232240e-04, 3.839060e-04, 3.445729e-04,
  3.052263e-04, 2.658677e-04, 2.264986e-04, 1.871207e-04, 1.477358e-04, 1.083460e-04,
  6.895707e-05, 2.962364e-05
};

List normixEM(const arma::vec& x, int m, int maxiter = 100, double tol = 1e-8) {
  int n = x.n_elem;
  vec probs = regspace(0.0, 1.0 / m, 1.0);
  vec q = quantile(x, probs);
  
  mat z = zeros<mat>(n, m);
  for (int i = 0; i < n; ++i) {
    int idx = m - 1;
    for (int j = 0; j < m; ++j) {
      if (x[i] <= q[j + 1]) {
        idx = j;
        break;
      }
    }
    z(i, idx) = 1.0;
  }
  
  double l_fac = -n * (std::log(n) + std::log(2.0 * M_PI) / 2.0);
  double llh = -datum::inf;
  vec n_j(m), mu(m);
  
  for (int it = 1; it <= maxiter; ++it) {
    n_j = trans(sum(z, 0));
    mu = (trans(z) * x) / n_j;
    
    double current_llh_sum = 0.0;
    
    for (int i = 0; i < n; ++i) {
      double xi = x[i];
      double pr_xi = 0.0;
      rowvec row_fx(m);
      
      for (int j = 0; j < m; ++j) {
        double diff = xi - mu[j];
        double val = n_j[j] * std::exp(-0.5 * diff * diff);
        row_fx[j] = val;
        pr_xi += val;
      }
      
      current_llh_sum += std::log(pr_xi);
      z.row(i) = row_fx / pr_xi;
    }
    
    double llh_old = llh;
    llh = current_llh_sum;
    if (std::abs(llh - llh_old) / (std::abs(llh) + 1e-7) <= tol) break;
  }
  
  double final_loglik = llh + l_fac;
  int n_par = 2 * m - 1; 
  double bic = -2.0 * final_loglik + n_par * std::log(n);
  
  return List::create(
    Named("mu") = mu, 
    Named("w") = n_j / n, 
    Named("loglik") = final_loglik, 
    Named("bic") = bic, 
    Named("m") = m
  );
}

List normix(arma::vec x, int max_m = 9, int maxiter = 100, double tol = 1.490116e-08) {
  if (max_m < 1) stop("max_m must be at least 1");
  
  List best_model = normixEM(x, 1, maxiter, tol);
  double min_bic = best_model["bic"];
  
  for (int m = 2; m <= max_m; ++m) {
    List current_model = normixEM(x, m, maxiter, tol);
    double current_bic = current_model["bic"];
    if (current_bic > min_bic) break;
    min_bic = current_bic;
    best_model = current_model;
  }
  return best_model;
}

// [[Rcpp::export]]
List plugin_mleb(arma::vec X, int max_m = 9) {
  int N = X.n_elem;
  
  List fit = normix(X, max_m);
  vec w = fit["w"];
  vec mu = fit["mu"];
  int m = fit["m"];
  
  double L = X.min();
  double U = X.max();
  double half_range = (U - L) / 2.0;
  double mid_range = (U + L) / 2.0;
  
  const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);
  double A1 = 0.0, A2 = 0.0, A3 = 0.0;
  
  for (int k = 0; k < 500; ++k) {
    double pt_k = LEGENDRE_X[k] * half_range + mid_range;
    double wt_k = LEGENDRE_W[k] * half_range;
    
    double fk = 0.0, f1k = 0.0, f2k = 0.0, f3k = 0.0;
    
    for (int j = 0; j < m; ++j) {
      double diff = mu[j] - pt_k;
      double pdf_val = inv_sqrt_2pi * std::exp(-0.5 * diff * diff);
      double f_j = pdf_val * w[j];
      
      fk  += f_j;
      f1k += diff * f_j;
      f2k += (diff * diff - 1.0) * f_j;
      f3k += (-3.0 * diff + diff * diff * diff) * f_j;
    }
    
    double term_k = f3k - (f2k * f1k) / fk;
    
    A1 += (term_k * term_k / fk) * wt_k;
    A2 += ((term_k * f1k) / fk) * wt_k;
    A3 += (f1k * f1k / fk) * wt_k;
  }
  
  double kappa = 1.0 / (4.0 * std::sqrt(M_PI));
  double B = U - L;
  double sd_X = std::sqrt(var(X));
  
  double term_k = (3.0 * kappa * B) / (std::pow(std::sqrt(6.0) * sd_X, 7.0) * N);
  double K = std::max(A1 - (A2 * A2) / A3, term_k);
  
  double b = std::pow(3.0 * kappa * B / K, 1.0 / 7.0) * std::pow(N, -1.0 / 7.0);
  double s = -A2 / (2.0 * A3) * b * b;
  
  return List::create(
    Named("b") = b,
    Named("s") = s,
    Named("m") = m,
    Named("mu") = mu,
    Named("w") = w,
    Named("A1") = A1,
    Named("A2") = A2,
    Named("A3") = A3
  );
}

// ============================================================================
// SECTION 2: SURE Parameter Optimization (Exact)
// ============================================================================

struct SUREData {
  const NumericVector& X;
  int N;
};

void eval_sure(double b, double s, const NumericVector& X, int N, double& risk, double* grad) {
  double inv_b = 1.0 / b;
  double inv_b2 = inv_b * inv_b;
  double inv_b3 = inv_b2 * inv_b;
  double lambda = (1.0 + s) * inv_b2;
  double coef_X = 1.0 - lambda;
  double dnorm0 = 1.0 / std::sqrt(2.0 * M_PI);
  
  double total_sq_diff = 0.0;
  double total_dmuhat = 0.0;
  double dRisk_db = 0.0;
  double dRisk_ds = 0.0;
  
  for (int j = 0; j < N; ++j) {
    double Xj = X[j];
    double Xj2 = Xj * Xj;
    
    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
    for (int i = 0; i < N; ++i) {
      double Xi = X[i];
      double diff = (Xi - Xj) * inv_b;
      double w = dnorm0 * std::exp(-0.5 * diff * diff);
      
      t0 += w; t1 += Xi * w; t2 += Xi * Xi * w;
      t3 += Xi * Xi * Xi * w; t4 += Xi * Xi * Xi * Xi * w;
    }
    
    double m1 = t1 / t0;
    double t2_over_t0 = t2 / t0;
    double v = t2_over_t0 - m1 * m1;
    
    double mu_j = coef_X * Xj + lambda * m1;
    double dmu_j = coef_X + lambda * (dnorm0 / t0 + v * inv_b2);
    double residual = mu_j - Xj;
    
    total_sq_diff += residual * residual;
    total_dmuhat += dmu_j;
    
    if (grad) {
      double dmu_ds = (m1 - Xj) * inv_b2;
      double ddmu_ds = inv_b2 * (dnorm0 / t0 + v * inv_b2 - 1.0);
      dRisk_ds += 2.0 * residual * dmu_ds + 2.0 * ddmu_ds;
      
      double dt0_db = inv_b3 * (t2 - 2.0 * Xj * t1 + Xj2 * t0);
      double dt1_db = inv_b3 * (t3 - 2.0 * Xj * t2 + Xj2 * t1);
      double dt2_db = inv_b3 * (t4 - 2.0 * Xj * t3 + Xj2 * t2);
      
      double dm1_db = (dt1_db - m1 * dt0_db) / t0;
      double dv_db = (dt2_db - t2_over_t0 * dt0_db) / t0 - 2.0 * m1 * dm1_db;
      
      double dmu_db = (2.0 * lambda * inv_b) * (Xj - m1) + lambda * dm1_db;
      double term1 = 2.0 * lambda * inv_b * (1.0 - dnorm0 / t0 - v * inv_b2);
      double term2 = lambda * (-dnorm0 * dt0_db / (t0 * t0));
      double term3 = lambda * inv_b2 * (dv_db - 2.0 * v * inv_b);
      double ddmu_db = term1 + term2 + term3;
      
      dRisk_db += 2.0 * residual * dmu_db + 2.0 * ddmu_db;
    }
  }
  
  risk = (total_sq_diff / N) + 2.0 * (total_dmuhat / N);
  if (grad) {
    grad[0] = dRisk_db / N;
    grad[1] = dRisk_ds / N;
  }
}

static double lbfgsb_fmin(int n, double* par, void* ex) {
  SUREData* data = static_cast<SUREData*>(ex);
  double risk;
  eval_sure(par[0], par[1], data->X, data->N, risk, NULL);
  return risk;
}

static void lbfgsb_gr(int n, double* par, double* gr, void* ex) {
  SUREData* data = static_cast<SUREData*>(ex);
  double risk;
  eval_sure(par[0], par[1], data->X, data->N, risk, gr);
}

// [[Rcpp::export]]
List optimize_SURE_lbfgsb(const NumericVector& X, 
                          int max_iters = 100, double factr = 1e7, double pgtol = 1e-5) {
  int N = X.size();
  
  double h = 1.0 / std::sqrt(std::log(static_cast<double>(N)));
  double b_init = 2.0 * h;
  double s_init = 0.5;
  
  double par[2] = {b_init, s_init};
  
  double var_X = Rcpp::var(X);
  double b_upper = std::sqrt(6.0 * var_X);
  double s_upper = 6.0;
  
  double lower_bounds[2] = {1e-5, 0.0};
  double upper_bounds[2] = {b_upper, s_upper};
  int nbd[2] = {2, 2};
  
  SUREData data = {X, N};
  
  double Fmin = 0.0;
  int fail = 0, fncount = 0, grcount = 0;
  int m = 5, npar = 2;
  int trace = 0, nREPORT = 10;
  char msg[1024] = {0};
  
  lbfgsb(npar, m, par, lower_bounds, upper_bounds, nbd,
         &Fmin, &lbfgsb_fmin, &lbfgsb_gr, &fail, static_cast<void*>(&data),
         factr, pgtol, &fncount, &grcount, max_iters,
         msg, trace, nREPORT);
  
  double final_grad[2];
  double dummy_risk;
  eval_sure(par[0], par[1], X, N, dummy_risk, final_grad);
  
  return List::create(
    Named("b_opt")     = par[0],
                            Named("s_opt")     = par[1],
                                                    Named("min_risk")  = Fmin,
                                                    Named("grad")      = NumericVector::create(Named("b") = final_grad[0], Named("s") = final_grad[1]),
                                                    Named("fail")      = fail,
                                                    Named("fn_counts") = fncount,
                                                    Named("gr_counts") = grcount,
                                                    Named("message")   = std::string(msg)
  );
}

// ============================================================================
// SECTION 3: Kernel Tweedie Evaluation & Linear-Time PAVA
// ============================================================================

NumericVector pava(const NumericVector& y) {
  int n = y.size();
  std::vector<double> val(n);
  std::vector<double> w(n, 1.0);
  std::vector<int> len(n, 1);
  
  for (int i = 0; i < n; ++i) {
    val[i] = y[i];
  }
  
  int k = 0;
  for (int i = 0; i < n; ++i) {
    k++;
    val[k - 1] = y[i];
    w[k - 1] = 1.0;
    len[k - 1] = 1;
    
    while (k > 1 && val[k - 2] > val[k - 1]) {
      double new_w = w[k - 2] + w[k - 1];
      val[k - 2] = (w[k - 2] * val[k - 2] + w[k - 1] * val[k - 1]) / new_w;
      w[k - 2] = new_w;
      len[k - 2] += len[k - 1];
      k--;
    }
  }
  
  NumericVector fitted(n);
  int idx = 0;
  for (int j = 0; j < k; ++j) {
    for (int L = 0; L < len[j]; ++L) {
      fitted[idx++] = val[j];
    }
  }
  
  return fitted;
}

// Exact O(N^2) Tweedie Evaluation
// [[Rcpp::export]]
List kde_tweedie(const NumericVector& X, double b, double s) {
  int n = X.size();
  NumericVector tweedie(n);
  
  double inv_b = 1.0 / b;
  double inv_b2 = inv_b * inv_b;
  double dnorm0 = 1.0 / std::sqrt(2.0 * M_PI);
  double const_factor = dnorm0 * inv_b / n;
  
  for (int j = 0; j < n; ++j) {
    double Xj = X[j];
    
    double sum_f = 0.0;
    double sum_d1f = 0.0;
    
    for (int i = 0; i < n; ++i) {
      double diff = X[i] - Xj;
      double scaled_diff = diff * inv_b;
      double w = const_factor * std::exp(-0.5 * scaled_diff * scaled_diff);
      
      sum_f += w;
      sum_d1f += diff * inv_b2 * w;
    }
    
    tweedie[j] = Xj + (1.0 + s) * (sum_d1f / sum_f);
  }
  
  NumericVector mono_tweedie = pava(tweedie);
  
  return List::create(
    Named("tweedie") = tweedie,
    Named("mono_tweedie") = mono_tweedie
  );
}

// ============================================================================
// SECTION 4: Binned Kernel Tweedie Evaluation & Binned SURE
// ============================================================================

// [[Rcpp::export]]
List kde_tweedie_binned(const NumericVector& X, double b, double s, int M = 1024, double pad_ratio = 0.05) {
  int n = X.size();
  
  // 1. Set up grid with fixed padding based on the data range
  double min_x = X[0];
  double max_x = X[0];
  for (int i = 1; i < n; ++i) {
    if (X[i] < min_x) min_x = X[i];
    if (X[i] > max_x) max_x = X[i];
  }
  
  double data_range = max_x - min_x;
  double pad = (data_range > 0) ? (pad_ratio * data_range) : 1.0; // Fixed padding
  
  double g_min = min_x - pad;
  double g_max = max_x + pad;
  double delta = (g_max - g_min) / (M - 1);
  
  NumericVector g(M);
  for (int l = 0; l < M; ++l) {
    g[l] = g_min + l * delta;
  }
  
  // 2. Linear binning weights
  NumericVector W(M);
  for (int i = 0; i < n; ++i) {
    double rem = (X[i] - g_min) / delta;
    int k = static_cast<int>(std::floor(rem));
    
    if (k < 0) {
      W[0] += 1.0;
    } else if (k >= M - 1) {
      W[M - 1] += 1.0;
    } else {
      double fraction = rem - k;
      W[k]     += (1.0 - fraction);
      W[k + 1] += fraction;
    }
  }
  
  // 3. Grid-based KDE and derivative evaluation
  double inv_b = 1.0 / b;
  double inv_b2 = inv_b * inv_b;
  double dnorm0 = 1.0 / std::sqrt(2.0 * M_PI);
  double const_factor = dnorm0 * inv_b / n;
  
  NumericVector mu_grid(M);
  
  for (int l = 0; l < M; ++l) {
    double gl = g[l];
    double sum_f = 0.0;
    double sum_d1f = 0.0;
    
    for (int m = 0; m < M; ++m) {
      if (W[m] == 0.0) continue;
      
      double diff = gl - g[m];
      double scaled_diff = diff * inv_b;
      double w = const_factor * std::exp(-0.5 * scaled_diff * scaled_diff);
      
      sum_f += W[m] * w;
      sum_d1f += W[m] * (-diff * inv_b2) * w;
    }
    
    if (sum_f > 1e-12) {
      mu_grid[l] = gl + (1.0 + s) * (sum_d1f / sum_f);
    } else {
      mu_grid[l] = gl;
    }
  }
  
  // 4. Linear interpolation back to original sample points
  NumericVector tweedie(n);
  for (int j = 0; j < n; ++j) {
    double rem = (X[j] - g_min) / delta;
    int k = static_cast<int>(std::floor(rem));
    
    if (k < 0) {
      tweedie[j] = mu_grid[0];
    } else if (k >= M - 1) {
      tweedie[j] = mu_grid[M - 1];
    } else {
      double fraction = rem - k;
      tweedie[j] = (1.0 - fraction) * mu_grid[k] + fraction * mu_grid[k + 1];
    }
  }
  
  // 5. Apply PAVA monotonicity
  NumericVector mono_tweedie = pava(tweedie);
  
  return List::create(
    Named("tweedie") = tweedie,
    Named("mono_tweedie") = mono_tweedie
  );
}

// Binned SURE Evaluation & Binned Parameter Optimization

struct SUREBinnedData {
  int N;
  int M;
  double g_min;
  double delta;
  const NumericVector& W;
  const NumericVector& g;
};

void eval_sure_binned(double b, double s, const SUREBinnedData& data, double& risk, double* grad) {
  int M = data.M;
  int N = data.N;
  double inv_b = 1.0 / b;
  double inv_b2 = inv_b * inv_b;
  double inv_b3 = inv_b2 * inv_b;
  double lambda = (1.0 + s) * inv_b2;
  double coef_X = 1.0 - lambda;
  double dnorm0 = 1.0 / std::sqrt(2.0 * M_PI);
  
  NumericVector mu_grid(M);
  NumericVector dmu_grid(M);
  NumericVector dmu_db_grid(M);
  NumericVector dmu_ds_grid(M);
  NumericVector ddmu_db_grid(M);
  NumericVector ddmu_ds_grid(M);
  
  for (int j = 0; j < M; ++j) {
    double gj = data.g[j];
    double gj2 = gj * gj;
    
    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
    for (int i = 0; i < M; ++i) {
      if (data.W[i] == 0.0) continue;
      double gi = data.g[i];
      double diff = (gi - gj) * inv_b;
      double w = data.W[i] * dnorm0 * std::exp(-0.5 * diff * diff);
      
      t0 += w;
      t1 += gi * w;
      t2 += gi * gi * w;
      t3 += gi * gi * gi * w;
      t4 += gi * gi * gi * gi * w;
    }
    
    if (t0 < 1e-12) {
      mu_grid[j] = gj;
      dmu_grid[j] = 1.0;
      continue;
    }
    
    double m1 = t1 / t0;
    double t2_over_t0 = t2 / t0;
    double v = t2_over_t0 - m1 * m1;
    
    mu_grid[j] = coef_X * gj + lambda * m1;
    dmu_grid[j] = coef_X + lambda * (dnorm0 / t0 + v * inv_b2);

    if (grad) {
      dmu_ds_grid[j] = (m1 - gj) * inv_b2;
      ddmu_ds_grid[j] = inv_b2 * (dnorm0 / t0 + v * inv_b2 - 1.0);
      
      double dt0_db = inv_b3 * (t2 - 2.0 * gj * t1 + gj2 * t0);
      double dt1_db = inv_b3 * (t3 - 2.0 * gj * t2 + gj2 * t1);
      double dt2_db = inv_b3 * (t4 - 2.0 * gj * t3 + gj2 * t2);
      
      double dm1_db = (dt1_db - m1 * dt0_db) / t0;
      double dv_db = (dt2_db - t2_over_t0 * dt0_db) / t0 - 2.0 * m1 * dm1_db;
      
      dmu_db_grid[j] = (2.0 * lambda * inv_b) * (gj - m1) + lambda * dm1_db;
      double term1 = 2.0 * lambda * inv_b * (1.0 - dnorm0 / t0 - v * inv_b2);
      double term2 = lambda * (-dnorm0 * dt0_db / (t0 * t0));
      double term3 = lambda * inv_b2 * (dv_db - 2.0 * v * inv_b);
      ddmu_db_grid[j] = term1 + term2 + term3;
    }
  }
  
  double total_sq_diff = 0.0;
  double total_dmuhat = 0.0;
  double dRisk_db = 0.0;
  double dRisk_ds = 0.0;
  
  for (int j = 0; j < M; ++j) {
    if (data.W[j] == 0.0) continue;
    double w_j = data.W[j];
    double gj = data.g[j];
    double residual = mu_grid[j] - gj;
    
    total_sq_diff += w_j * (residual * residual);
    total_dmuhat += w_j * dmu_grid[j];
    
    if (grad) {
      dRisk_ds += w_j * (2.0 * residual * dmu_ds_grid[j] + 2.0 * ddmu_ds_grid[j]);
      dRisk_db += w_j * (2.0 * residual * dmu_db_grid[j] + 2.0 * ddmu_db_grid[j]);
    }
  }
  
  risk = (total_sq_diff / N) + 2.0 * (total_dmuhat / N);
  if (grad) {
    grad[0] = dRisk_db / N;
    grad[1] = dRisk_ds / N;
  }
}

static double lbfgsb_fmin_binned(int n, double* par, void* ex) {
  SUREBinnedData* data = static_cast<SUREBinnedData*>(ex);
  double risk;
  eval_sure_binned(par[0], par[1], *data, risk, NULL);
  return risk;
}

static void lbfgsb_gr_binned(int n, double* par, double* gr, void* ex) {
  SUREBinnedData* data = static_cast<SUREBinnedData*>(ex);
  double risk;
  eval_sure_binned(par[0], par[1], *data, risk, gr);
}

// [[Rcpp::export]]
List optimize_SURE_binned_lbfgsb(const NumericVector& X, int M = 1024, double pad_ratio = 0.05,
                                 int max_iters = 100, double factr = 1e7, double pgtol = 1e-5) {
  int N = X.size();
  
  // 1. Grid setup and binning
  double min_x = X[0];
  double max_x = X[0];
  for (int i = 1; i < N; ++i) {
    if (X[i] < min_x) min_x = X[i];
    if (X[i] > max_x) max_x = X[i];
  }
  
  double data_range = max_x - min_x;
  double pad = (data_range > 0) ? (pad_ratio * data_range) : 1.0;
  
  double g_min = min_x - pad;
  double g_max = max_x + pad;
  double delta = (g_max - g_min) / (M - 1);
  
  NumericVector g(M);
  for (int l = 0; l < M; ++l) {
    g[l] = g_min + l * delta;
  }
  
  NumericVector W(M);
  for (int i = 0; i < N; ++i) {
    double rem = (X[i] - g_min) / delta;
    int k = static_cast<int>(std::floor(rem));
    
    if (k < 0) {
      W[0] += 1.0;
    } else if (k >= M - 1) {
      W[M - 1] += 1.0;
    } else {
      double fraction = rem - k;
      W[k]     += (1.0 - fraction);
      W[k + 1] += fraction;
    }
  }
  
  // 2. Initialization and bounds
  double h = 1.0 / std::sqrt(std::log(static_cast<double>(N)));
  double b_init = 2.0 * h;
  double s_init = 0.5;
  double par[2] = {b_init, s_init};
  
  double var_X = Rcpp::var(X);
  double b_upper = std::sqrt(6.0 * var_X);
  double s_upper = 6.0;
  
  double lower_bounds[2] = {2.0 * delta, 0.0};
  double upper_bounds[2] = {b_upper, s_upper};
  int nbd[2] = {2, 2};
  
  SUREBinnedData data = {N, M, g_min, delta, W, g};
  
  double Fmin = 0.0;
  int fail = 0, fncount = 0, grcount = 0;
  int m_lbfgs = 5, npar = 2;
  int trace = 0, nREPORT = 10;
  char msg[1024] = {0};
  
  lbfgsb(npar, m_lbfgs, par, lower_bounds, upper_bounds, nbd,
         &Fmin, &lbfgsb_fmin_binned, &lbfgsb_gr_binned, &fail, static_cast<void*>(&data),
         factr, pgtol, &fncount, &grcount, max_iters,
         msg, trace, nREPORT);
  
  double final_grad[2];
  double dummy_risk;
  eval_sure_binned(par[0], par[1], data, dummy_risk, final_grad);
  
  return List::create(
    Named("b_opt")     = par[0],
                            Named("s_opt")     = par[1],
                                                    Named("min_risk")  = Fmin,
                                                    Named("grad")      = NumericVector::create(Named("b") = final_grad[0], Named("s") = final_grad[1]),
                                                    Named("fail")      = fail,
                                                    Named("fn_counts") = fncount,
                                                    Named("gr_counts") = grcount,
                                                    Named("message")   = std::string(msg)
  );
}