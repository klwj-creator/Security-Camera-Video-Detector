import unittest

from cctv_detector.evidence import Confirmation, ConfirmationConfig


class ConfirmationTests(unittest.TestCase):
    def test_repeated_evidence_alerts_once(self):
        engine = Confirmation(ConfirmationConfig())
        results = [engine.observe(7, t, True) for t in (0, 0.15, 0.3, 0.45, 0.6)]
        self.assertEqual(results, [False, False, True, False, False])

    def test_isolated_hits_do_not_accumulate_forever(self):
        engine = Confirmation(ConfirmationConfig())
        self.assertFalse(any(engine.observe(1, t, True) for t in (0, 1.5, 3, 4.5)))

    def test_negative_observations_count_against_confirmation(self):
        engine = Confirmation(ConfirmationConfig(min_ratio=0.8))
        results = [engine.observe(1, i * 0.1, hit)
                   for i, hit in enumerate((True, False, True, False, True, False))]
        self.assertFalse(any(results))

    def test_short_burst_and_duplicate_timestamp_cannot_confirm(self):
        engine = Confirmation(ConfirmationConfig())
        self.assertFalse(engine.observe(1, 0, True))
        self.assertFalse(engine.observe(1, 0, True))
        self.assertFalse(engine.observe(1, 0.01, True))
        self.assertFalse(engine.observe(1, 0.02, True))
        self.assertTrue(engine.observe(1, 0.3, True))

    def test_expired_track_id_does_not_inherit_evidence(self):
        engine = Confirmation(ConfirmationConfig(track_ttl=0.5))
        engine.observe(1, 0, True)
        engine.observe(1, 0.2, True)
        engine.advance(1)
        self.assertFalse(engine.observe(1, 1, True))
        self.assertEqual(len(engine.tracks[1].samples), 1)

    def test_tracks_are_independent_and_expired_states_are_removed(self):
        engine = Confirmation(ConfirmationConfig())
        for i in range(3):
            self.assertFalse(engine.observe(i, i * 0.1, True))
        engine.advance(5)
        self.assertEqual(engine.tracks, {})

    def test_invalid_times_and_settings_are_rejected(self):
        engine = Confirmation(ConfirmationConfig())
        engine.advance(3)
        for timestamp in (2, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                engine.advance(timestamp)
        for settings in ({"min_hits": 0}, {"min_ratio": 1.1}, {"min_span": 3},
                         {"window_seconds": float("nan")}, {"track_ttl": 0}):
            with self.assertRaises(ValueError):
                ConfirmationConfig(**settings)


if __name__ == "__main__":
    unittest.main()
