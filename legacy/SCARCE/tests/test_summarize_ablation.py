import csv
import os
import tempfile
import unittest

from summarize_ablation import EXPERIMENTS, build_summary


class SummarizeAblationTest(unittest.TestCase):
    def test_matched_seed_summary(self):
        with tempfile.TemporaryDirectory() as result_dir:
            method_values = (
                (50.0, 52.0),
                (45.0, 47.0),
                (40.0, 42.0),
                (30.0, 32.0),
            )
            for (_, pattern), values in zip(EXPERIMENTS, method_values):
                for seed, value in enumerate(values):
                    relative_path = pattern.replace('*', str(seed))
                    path = os.path.join(result_dir, relative_path)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, 'w', newline='') as file_handle:
                        writer = csv.writer(file_handle)
                        writer.writerow(('run_idx', 'acc', 'std'))
                        writer.writerow(('in total', value, 0.0))

            rows = build_summary(result_dir, required_seeds=(0, 1))

        self.assertEqual(rows[0]['mean_accuracy'], 51.0)
        self.assertEqual(rows[1]['mean_accuracy'], 46.0)
        self.assertEqual(rows[1]['delta_vs_adiw'], -5.0)
        self.assertEqual(rows[1]['paired_delta_std'], 0.0)
        self.assertEqual(rows[2]['mean_accuracy'], 41.0)
        self.assertEqual(rows[3]['mean_accuracy'], 31.0)

    def test_missing_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as result_dir:
            with self.assertRaisesRegex(ValueError, 'missing seeds'):
                build_summary(result_dir, required_seeds=(0,))


if __name__ == '__main__':
    unittest.main()
