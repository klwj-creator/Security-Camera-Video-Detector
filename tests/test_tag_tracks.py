import unittest

from cctv_detector.tag_tracks import associate_body, build_trajectories


def tag(x,y):
    return {"tag_box":[x-2,y-2,x+2,y+2],"score":0.9,"person_box":[0,0,100,100]}


class TagTrajectoryTests(unittest.TestCase):
    def test_short_gap_is_interpolated_without_extrapolation(self):
        frames = [[],[tag(10,10)],[],[],[tag(22,10)],[]]
        result = build_trajectories(frames,100,25,max_gap=3)
        self.assertEqual([bool(p) for p in result],[False,True,True,True,True,False])
        self.assertEqual(result[2][0]["evidence"],"interpolated")
        self.assertIsNone(result[2][0]["score"])
        self.assertEqual(result[4][0]["evidence"],"direct")

    def test_isolated_or_distant_detections_do_not_confirm(self):
        self.assertEqual(build_trajectories([[tag(10,10)],[],[tag(95,95)]],100,25),[[],[],[]])

    def test_long_gap_cannot_link_separate_appearances(self):
        frames = [[tag(10,10)]] + [[] for _ in range(10)] + [[tag(10,10)]]
        self.assertFalse(any(build_trajectories(frames,100,25,max_gap=3)))

    def test_two_separate_tags_keep_separate_short_tracks(self):
        rows = build_trajectories([[tag(10,10),tag(80,80)],[tag(12,10),tag(82,80)]],100,25)
        self.assertEqual(len(rows[0]),2)
        self.assertEqual({p["track_id"] for p in rows[0]}, {p["track_id"] for p in rows[1]})

    def test_background_tag_and_oversized_pattern_are_rejected(self):
        people = [[0,0,100,100,0.9,0]]
        self.assertIsNone(associate_body([110,110,120,120],people))
        self.assertIsNone(associate_body([0,0,90,90],people))
        self.assertEqual(associate_body([20,20,30,30],people),[0,0,100,100])


if __name__ == "__main__":
    unittest.main()
