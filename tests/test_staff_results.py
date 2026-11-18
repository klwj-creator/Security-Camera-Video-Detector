import unittest

from cctv_detector.staff_results import select_tag_owners, summarize


def person(track_id=1, score=0.95, center=(50, 50), box=(0, 0, 100, 100)):
    return {"track_id": track_id, "box": box,
            "tag": {"center": center, "score": score, "contrast": 30}}


class StaffResultsTests(unittest.TestCase):
    def test_tag_match_does_not_label_a_long_track_history(self):
        rows = [{"frame": 0, "people": [person(score=0.1)]},
                {"frame": 90, "people": [person(score=0.1)]},
                {"frame": 100, "people": [person()]},
                {"frame": 102, "people": [person()]}]
        result = summarize(rows, 25)
        self.assertEqual([r["status"] for r in result],
                         ["not_identified", "track_inferred", "direct_tag_evidence", "direct_tag_evidence"])

    def test_isolated_high_score_is_insufficient(self):
        result = summarize([{"frame": 10, "people": [person()]}], 25)
        self.assertEqual(result[0]["people"], [])

    def test_one_tag_has_one_owner_when_person_boxes_overlap(self):
        near = person(1, center=(50, 50))
        far = person(2, center=(51, 50), box=(40, 0, 140, 100))
        self.assertEqual([p["track_id"] for p in select_tag_owners([far, near])], [1])

    def test_unsupported_frames_stay_unidentified(self):
        rows = [{"frame": 100, "people": [person()]}, {"frame": 102, "people": [person()]},
                {"frame": 103, "people": []}, {"frame": 130, "people": [person(score=0.1)]}]
        result = summarize(rows, 25)
        self.assertEqual(result[2]["status"], "not_identified")
        self.assertEqual(result[3]["status"], "not_identified")


if __name__ == "__main__":
    unittest.main()
