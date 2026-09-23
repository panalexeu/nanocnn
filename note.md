Dropping the batch dim entirely. Input is `(Cin, Hin, Win)`, output is `(Cout, Hout, Wout)`:

```c
for (j = 0; j < Cout; j++)                 // which output map
  for (h = 0; h < Hout; h++)               // output row
    for (w = 0; w < Wout; w++) {           // output col
      float s = bias[j];
      for (k = 0; k < Cin; k++)            // input map   -- SUMMED
        for (y = 0; y < kH; y++)           // kernel row  -- SUMMED
          for (x = 0; x < kW; x++)         // kernel col  -- SUMMED
            s += weight[j][k][y][x] * input[k][h*stride + y][w*stride + x];
      out[j][h][w] = s;
    }
```

Six loops. The split is what matters:

The outer three (`j, h, w`) each **write a distinct output element**. They are the shape of the result. One index per dimension of `out`.

The inner three (`k, y, x`) each **accumulate into `s`**. They vanish. They are the shape of one weight block.

That's the entire operation. Everything else is bookkeeping about the bounds:

- `Hout = (Hin + 2*pad - kH) / stride + 1`, same for `Wout`.
- `weight` is `[Cout][Cin][kH][kW]` — the four indices used, in that order.
- `bias` is `[Cout]` — indexed by `j` only, which is why it's constant across `h, w`.

Note which loop variables index which array. `weight` uses `j, k, y, x` and never `h, w`: one weight block per output map, reused at every position. `input` uses `k` and the strided spatial expression, never `j`: every output map reads the same input. That pair of facts is weight sharing and multi-channel mixing respectively.

For LeCun's H2: `Cout=12, Cin=8, kH=kW=5, stride=2, Hout=Wout=4`. Inner loops run 8·5·5 = 200 multiply-adds per output element, which is the "200 inputs, 200 weights" from the paper. Outer loops run 12·4·4 = 192 times, so 192 output units total.