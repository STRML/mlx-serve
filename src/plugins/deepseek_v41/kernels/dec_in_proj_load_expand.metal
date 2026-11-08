// The stream a launch reads: the hc expand of `x` by the previous block's summed output `y` and its split, kept in `x1`.
#define DSV41_STREAM8(R_, V_) dsv41_expand8(x, y, spost, scomb, R_, 8 * int(t), D, V_)
#define DSV41_KEEP8(R_, V_) dsv41_keep8(x1, R_, 8 * int(t), D, V_)
