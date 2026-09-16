import csv
import os
import tempfile
import unittest

from summarize_shift_ablation import CONDITIONS, build_shift_summary, patterns


class SummarizeShiftAblationTest(unittest.TestCase):
    def test_four_condition_paired_summary(self):
        with tempfile.TemporaryDirectory() as result_dir:
            for _, shift in CONDITIONS:
                adiw_pattern, pooled_pattern = patterns(shift)
                for pattern, values in (
                        (adiw_pattern, (60.0, 63.0)),
                        (pooled_pattern, (58.0, 60.0))):
                    for seed, value in enumerate(values):
                        path = os.path.join(
                            result_dir, pattern.replace('*', str(seed))
                        )
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, 'w', newline='') as file_handle:
                            writer = csv.writer(file_handle)
                            writer.writerow(('run_idx', 'acc', 'std'))
                            writer.writerow(('in total', value, 0.0))

            rows = build_shift_summary(result_dir, required_seeds=(0, 1))

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]['adiw_mean'], 61.5)
        self.assertEqual(rows[0]['pooled_mean'], 59.0)
        self.assertEqual(rows[0]['adiw_minus_pooled'], 2.5)


if __name__ == '__main__':
    unittest.main()
