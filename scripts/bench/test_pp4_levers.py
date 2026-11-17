"""Offline projections preserve non-read costs and measured decoder layers."""
import copy
import unittest

import pp4_levers as levers
import pp4_step0 as pp4


class LeverContract(unittest.TestCase):
    def test_only_read_intervals_scale_and_origin_is_request_arrival(self):
        events = [dict(kind='pp4_catalog', start_ns=10**9, ready_ns=3 * 10**9)]
        for layer, start, end, copied in ((1, 4, 8, 9), (14, 11, 17, 18)):
            events.append(dict(kind='pp4_bucket', layer=layer,
                read_start_ns=start * 10**9, read_end_ns=end * 10**9,
                copy_end_ns=copied * 10**9))
        original = copy.deepcopy(events)
        self.assertEqual(levers.scale_reads(events, (1, 1)), [[8, 17]])
        self.assertEqual(levers.scale_reads(events, (.5, .5)), [[6, 12]])
        self.assertEqual(events, original)

    def test_decoder_costs_are_measured_and_missing_duplicate_or_wrong_rows_refuse(self):
        events = [dict(kind='pp4_layer', p0=0, rows=2048, layer=i,
                       layer_rows=2048 if i < 20 else 128,
                       phase='encoder' if i < 20 else 'decoder', compute_us=1000)
                  for i in range(40)]
        events.append(dict(kind='pp4_chunk', p0=0, rows=2048, elapsed_us=45000,
            encoder_us=20000, compressor_us=1000, decoder_us=20000,
            dspark_us=1000, head_us=1000, x_us=22000, dspark_enabled=True))
        chunks = levers.decoder_chunks(events)
        self.assertEqual(chunks[0]['decoder_layer_s'], [.001] * 20)
        wrong_rows = copy.deepcopy(events)
        wrong_rows[20]['layer_rows'] = 2048
        for bad in (events[:20] + events[21:], events + [events[20]], wrong_rows):
            with self.subTest(bad=bad[20]), self.assertRaises(ValueError):
                levers.decoder_chunks(bad)

    def test_split_moves_compressor_with_layer20_and_retains_dspark_and_head(self):
        chunk = dict(layers_s=[.01] * 20, decoder_layer_s=[.002] * 20,
            overhead_s=.003, compressor_s=.004, decoder_s=.04,
            dspark_s=.005, head_s=.006, x_s=.049)
        costs = levers.cost_model([chunk], max, hop_s=.001)
        result = costs((8, 16, 24))
        self.assertEqual([round(v, 3) for v in result], [.084, .081, .053, .043])
        for cuts in ((2, 8, 14), (8, 14, 20)):
            expected = pp4.stage_costs(chunk['layers_s'], chunk['x_s'], '/'.join(map(str, cuts)),
                .001, chunk['overhead_s'], chunk['head_s'])
            for actual, cost in zip(costs(cuts), expected):
                self.assertAlmostEqual(actual, cost)

    def test_reader_curve_interpolates_throughput_and_anchors_identity(self):
        self.assertEqual(levers.read_factors(32, 32), (1, 1))
        factors = levers.read_factors(6, 16)
        self.assertAlmostEqual(factors[0], 33848 / 67802)
        self.assertAlmostEqual(factors[1], 33978.5 / 68304)


if __name__ == '__main__':
    unittest.main()
