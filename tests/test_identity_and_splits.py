import json
from collections import Counter
import math
import unittest
from itertools import chain, product

from flowfm.data.identity import canonical_json, make_sample_id, stable_hash
from flowfm.data.splits import (
    SplitManifest,
    deterministic_group_split,
    deterministic_stratified_group_split,
)


class IdentityTests(unittest.TestCase):
    def test_canonical_json_is_independent_of_mapping_insertion_order(self):
        left = {"label": "normal", "features": {"b": 2, "a": [1, True, None]}}
        right = {"features": {"a": [1, True, None], "b": 2}, "label": "normal"}

        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(
            canonical_json(left),
            '{"features":{"a":[1,true,null],"b":2},"label":"normal"}',
        )

    def test_canonical_json_uses_utf8_and_normalizes_tuples_as_arrays(self):
        self.assertEqual(
            canonical_json({"city": "上海", "ports": (80, 443)}),
            '{"city":"上海","ports":[80,443]}',
        )

    def test_canonical_json_rejects_unsupported_or_ambiguous_values(self):
        invalid_values = [
            {"value": math.nan},
            {"value": math.inf},
            {"value": -math.inf},
            {1: "non-string key"},
            {"value": {1, 2}},
            {"value": b"bytes"},
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    canonical_json(value)

    def test_stable_hash_is_lowercase_sha256_and_namespaced(self):
        value = {"a": 1, "b": 2}
        first = stable_hash(value, "dataset-a")

        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(first, stable_hash({"b": 2, "a": 1}, "dataset-a"))
        self.assertNotEqual(first, stable_hash(value, "dataset-b"))
        with self.assertRaises(ValueError):
            stable_hash(value, "")

    def test_sample_id_changes_with_each_identity_component(self):
        record = {"src": "10.0.0.1", "dst": "10.0.0.2", "bytes": 20}
        baseline = make_sample_id(record, "cicids2017", "v1")

        self.assertEqual(baseline, make_sample_id(dict(reversed(list(record.items()))), "cicids2017", "v1"))
        self.assertNotEqual(baseline, make_sample_id({**record, "bytes": 21}, "cicids2017", "v1"))
        self.assertNotEqual(baseline, make_sample_id(record, "unsw-nb15", "v1"))
        self.assertNotEqual(baseline, make_sample_id(record, "cicids2017", "v2"))
        for dataset_id, schema_version in (("", "v1"), ("cicids2017", "")):
            with self.assertRaises(ValueError):
                make_sample_id(record, dataset_id, schema_version)


class SplitManifestTests(unittest.TestCase):
    RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}

    @staticmethod
    def records(group_count=20):
        return [
            {"sample_id": f"sample-{group}-{member}", "group_id": f"group-{group}"}
            for group in range(group_count)
            for member in range(2)
        ]

    def make_manifest(self, records=None, seed=17):
        return deterministic_group_split(
            self.records() if records is None else records,
            ratios=self.RATIOS,
            seed=seed,
            schema_version="v1",
            source_hash="source-abc",
        )

    def test_entire_groups_are_assigned_to_one_split(self):
        manifest = self.make_manifest()

        seen_samples = set()
        for split_name, sample_ids in manifest.splits.items():
            for sample_id in sample_ids:
                self.assertNotIn(sample_id, seen_samples)
                seen_samples.add(sample_id)
                group_id = manifest.sample_groups[sample_id]
                self.assertEqual(manifest.group_assignments[group_id], split_name)
        self.assertEqual(seen_samples, {record["sample_id"] for record in self.records()})

    def test_same_inputs_are_deterministic_and_seed_can_change_assignment(self):
        first = self.make_manifest(seed=17)
        repeated = self.make_manifest(records=list(reversed(self.records())), seed=17)
        changed = self.make_manifest(seed=18)

        self.assertEqual(first, repeated)
        self.assertEqual(first.to_json(), repeated.to_json())
        self.assertNotEqual(first.group_assignments, changed.group_assignments)

    def test_json_round_trip_preserves_exact_content_and_is_deterministic(self):
        manifest = self.make_manifest()
        encoded = manifest.to_json()
        decoded = SplitManifest.from_json(encoded)

        self.assertEqual(decoded, manifest)
        self.assertEqual(decoded.to_json(), encoded)
        self.assertEqual(list(json.loads(encoded)), sorted(json.loads(encoded)))

    def test_manifest_collections_cannot_be_mutated_after_validation(self):
        manifest = self.make_manifest()

        with self.assertRaises(TypeError):
            manifest.splits["train"] = ()
        with self.assertRaises(AttributeError):
            manifest.splits["train"].append("injected")
        with self.assertRaises(TypeError):
            manifest.ratios["train"] = 1.0
        with self.assertRaises(TypeError):
            manifest.sample_groups["injected"] = "group-x"
        with self.assertRaises(TypeError):
            manifest.group_assignments["group-x"] = "test"

    def test_to_dict_returns_fresh_deep_copies(self):
        manifest = self.make_manifest()
        exported = manifest.to_dict()
        original_train = tuple(manifest.splits["train"])

        exported["splits"]["train"].append("injected")
        exported["ratios"]["train"] = 1.0
        exported["sample_groups"]["injected"] = "group-x"
        exported["group_assignments"]["group-x"] = "train"

        self.assertEqual(manifest.splits["train"], original_train)
        self.assertEqual(manifest.ratios["train"], 0.6)
        self.assertNotIn("injected", manifest.sample_groups)
        self.assertNotIn("group-x", manifest.group_assignments)

    def test_identical_duplicate_records_are_deduplicated(self):
        records = self.records(group_count=2)
        manifest = self.make_manifest(records=records + [dict(records[0])])

        assigned = list(chain.from_iterable(manifest.splits.values()))
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {record["sample_id"] for record in records})

    def test_conflicting_duplicate_sample_group_is_rejected(self):
        records = [
            {"sample_id": "same", "group_id": "group-a"},
            {"sample_id": "same", "group_id": "group-b"},
        ]

        with self.assertRaises(ValueError):
            self.make_manifest(records=records)

    def test_duplicate_sample_and_group_with_different_content_is_rejected(self):
        records = [
            {"sample_id": "same", "group_id": "group-a", "label": "normal"},
            {"group_id": "group-a", "label": "attack", "sample_id": "same"},
        ]

        with self.assertRaisesRegex(ValueError, "conflicting record content"):
            self.make_manifest(records=records)

    def test_equal_groups_realize_exact_requested_counts(self):
        records = [
            {"sample_id": f"sample-{index}", "group_id": f"group-{index}"}
            for index in range(5)
        ]

        manifest = self.make_manifest(records=records, seed=0)

        self.assertEqual(
            {name: len(sample_ids) for name, sample_ids in manifest.splits.items()},
            {"train": 3, "val": 1, "test": 1},
        )

    def test_unequal_groups_minimize_squared_target_deviation(self):
        group_sizes = {"oversized": 8, "small-a": 1, "small-b": 1}
        records = [
            {"sample_id": f"{group_id}-{index}", "group_id": group_id}
            for group_id, size in group_sizes.items()
            for index in range(size)
        ]

        manifest = self.make_manifest(records=records, seed=5)
        counts = {name: len(sample_ids) for name, sample_ids in manifest.splits.items()}
        targets = {name: len(records) * self.RATIOS[name] for name in self.RATIOS}
        actual_error = sum((counts[name] - targets[name]) ** 2 for name in self.RATIOS)
        possible_errors = []
        for assignments in product(self.RATIOS, repeat=len(group_sizes)):
            possible_counts = dict.fromkeys(self.RATIOS, 0)
            for (group_id, size), split_name in zip(group_sizes.items(), assignments):
                possible_counts[split_name] += size
            possible_errors.append(
                sum((possible_counts[name] - targets[name]) ** 2 for name in self.RATIOS)
            )

        self.assertEqual(actual_error, min(possible_errors))
        self.assertEqual(counts, {"train": 8, "val": 1, "test": 1})
        self.assertEqual(manifest.group_assignments["oversized"], "train")

    def test_ratio_contract_is_strict(self):
        invalid_ratios = [
            {"train": 0.8, "test": 0.2},
            {"train": 0.8, "val": 0.1, "test": 0.2},
            {"train": 0.0, "val": 0.5, "test": 0.5},
            {"train": 0.8, "val": -0.1, "test": 0.3},
            {"train": math.nan, "val": math.nan, "test": 0.0},

        ]

        for ratios in invalid_ratios:
            with self.subTest(ratios=ratios):
                with self.assertRaises(ValueError):
                    deterministic_group_split([], ratios, 1, "v1", "source")

    def test_empty_records_produce_valid_empty_manifest(self):
        manifest = self.make_manifest(records=[])

        self.assertEqual(manifest.splits, {"train": (), "val": (), "test": ()})
        self.assertEqual(manifest.sample_groups, {})
        self.assertEqual(manifest.group_assignments, {})
        self.assertEqual(SplitManifest.from_json(manifest.to_json()), manifest)

    def test_manifest_construction_and_load_validate_leakage(self):
        fields = dict(
            splits={"train": ["sample-a"], "val": ["sample-a"], "test": []},
            ratios=self.RATIOS,
            seed=1,
            schema_version="v1",
            source_hash="source",
            sample_groups={"sample-a": "group-a"},
            group_assignments={"group-a": "train"},
        )
        with self.assertRaises(ValueError):
            SplitManifest(**fields)
        with self.assertRaises(ValueError):
            SplitManifest.from_json(json.dumps(fields))

        crossed_group = dict(fields)
        crossed_group["splits"] = {"train": ["sample-a"], "val": ["sample-b"], "test": []}
        crossed_group["sample_groups"] = {"sample-a": "group-a", "sample-b": "group-a"}
        with self.assertRaises(ValueError):
            SplitManifest(**crossed_group)

    def test_json_load_rejects_missing_extra_and_duplicate_fields(self):
        data = self.make_manifest().to_dict()

        missing = dict(data)
        missing.pop("source_hash")
        with self.assertRaisesRegex(ValueError, "top-level fields"):
            SplitManifest.from_json(json.dumps(missing))

        extra = dict(data)
        extra["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "top-level fields"):
            SplitManifest.from_json(json.dumps(extra))

        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            SplitManifest.from_json('{"seed":1,"seed":2}')


class StratifiedGroupSplitTests(unittest.TestCase):
    RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}

    @staticmethod
    def records():
        return [
            {
                "sample_id": f"{label}-{index}",
                "group_id": f"{label}-group-{index}",
                "stratum": label,
            }
            for label in ("benign", "exploit", "rare")
            for index in range(5)
        ]

    def make_manifest(self, records=None, seed=31):
        return deterministic_stratified_group_split(
            self.records() if records is None else records,
            ratios=self.RATIOS,
            seed=seed,
            schema_version="schema-v1",
            source_hash="source-hash",
        )

    def test_realizes_exact_per_stratum_counts_when_groups_are_independent(self):
        records = self.records()
        manifest = self.make_manifest(records)
        sample_strata = {record["sample_id"]: record["stratum"] for record in records}

        counts = {
            split_name: Counter(sample_strata[sample_id] for sample_id in sample_ids)
            for split_name, sample_ids in manifest.splits.items()
        }

        self.assertEqual(counts["train"], Counter({"benign": 3, "exploit": 3, "rare": 3}))
        self.assertEqual(counts["val"], Counter({"benign": 1, "exploit": 1, "rare": 1}))
        self.assertEqual(counts["test"], Counter({"benign": 1, "exploit": 1, "rare": 1}))

    def test_mixed_label_group_is_never_split(self):
        records = [
            {"sample_id": "mixed-a", "group_id": "mixed", "stratum": "benign"},
            {"sample_id": "mixed-b", "group_id": "mixed", "stratum": "attack"},
            *self.records(),
        ]

        manifest = self.make_manifest(records)
        assigned = {
            sample_id: split_name
            for split_name, sample_ids in manifest.splits.items()
            for sample_id in sample_ids
        }

        self.assertEqual(assigned["mixed-a"], assigned["mixed-b"])

    def test_is_independent_of_input_order_and_seed_breaks_ties(self):
        records = self.records()
        first = self.make_manifest(records, seed=31)
        repeated = self.make_manifest(list(reversed(records)), seed=31)
        changed = self.make_manifest(records, seed=32)

        self.assertEqual(first.to_json(), repeated.to_json())
        self.assertNotEqual(first.group_assignments, changed.group_assignments)

    def test_requires_nonempty_string_stratum(self):
        for invalid in (
            {"sample_id": "a", "group_id": "g"},
            {"sample_id": "a", "group_id": "g", "stratum": ""},
            {"sample_id": "a", "group_id": "g", "stratum": 1},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.make_manifest([invalid])


if __name__ == "__main__":
    unittest.main()
