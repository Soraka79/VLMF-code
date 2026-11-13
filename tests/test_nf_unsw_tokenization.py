import unittest

from flowfm.data.contracts import FlowSample
from flowfm.data.identity import stable_hash
from flowfm.data.nf_unsw_dataset import NFUNSWDevelopmentSamples
from flowfm.data.nf_unsw_preprocessing import (
    TRAINING_SAMPLE_IDS_NAMESPACE,
    fit_nf_unsw_preprocessor,
)
from flowfm.data.nf_unsw_tokenization import (
    build_nf_unsw_tokenizer_metadata,
)
from flowfm.data.schemas.nf_unsw_nb15_v2 import (
    NF_UNSW_NB15_V2_SCHEMA,
)


def raw_row(*, source_port, protocol):
    row = {
        name: "0"
        for name in NF_UNSW_NB15_V2_SCHEMA.model_feature_names
    }
    row["L4_SRC_PORT"] = source_port
    row["L4_DST_PORT"] = "80"
    row["PROTOCOL"] = protocol
    return row


def make_sample(preprocessor, row, sample_id):
    encoded = preprocessor.transform(row)
    return FlowSample(
        features=encoded.features,
        feature_ids=encoded.feature_ids,
        categorical_mask=encoded.categorical_mask,
        missing_mask=encoded.missing_mask,
        category_tokens=encoded.category_tokens,
        binary_label="attack",
        coarse_label="dos",
        native_label="DoS",
        dataset_id=NF_UNSW_NB15_V2_SCHEMA.dataset_id,
        sample_id=sample_id,
        feature_key=f"feature-{sample_id}",
        conflict_flag=False,
        conflict_group_size=1,
        conflict_label_count=1,
        source_file="fixture.csv",
        source_row_id=sample_id.removeprefix("sample-"),
    )


class NFUNSWTokenizerMetadataTests(unittest.TestCase):
    def setUp(self):
        self.train_rows = (
            raw_row(source_port="1024", protocol="6"),
            raw_row(source_port="1300", protocol="17"),
        )
        self.preprocessor = fit_nf_unsw_preprocessor(
            self.train_rows
        )
        self.train_samples = tuple(
            make_sample(
                self.preprocessor,
                row,
                f"sample-{index}",
            )
            for index, row in enumerate(self.train_rows, start=1)
        )
        self.val_sample = make_sample(
            self.preprocessor,
            raw_row(source_port="65535", protocol="999"),
            "sample-3",
        )
        self.test_sample = make_sample(
            self.preprocessor,
            raw_row(source_port="54321", protocol="888"),
            "sample-4",
        )
        self.training_hash = stable_hash(
            [sample.sample_id for sample in self.train_samples],
            TRAINING_SAMPLE_IDS_NAMESPACE,
        )
        self.dataset = NFUNSWDevelopmentSamples(
            splits={
                "train": self.train_samples,
                "val": (self.val_sample,),
                "test": (self.test_sample,),
            },
            preprocessor_hash=self.preprocessor.preprocessor_hash,
            split_manifest_sha256="split-manifest-sha256",
            raw_csv_sha1="raw-csv-sha1",
            schema_version=NF_UNSW_NB15_V2_SCHEMA.version,
            training_sample_ids_hash=self.training_hash,
        )
        self.data_manifest = {
            "dataset_id": NF_UNSW_NB15_V2_SCHEMA.dataset_id,
            "manifest_sha256": "data-manifest-sha256",
        }

    def test_builds_train_only_main_model_metadata_and_preserves_unseen_ports(self):
        metadata = build_nf_unsw_tokenizer_metadata(
            self.dataset,
            self.preprocessor,
            data_manifest=self.data_manifest,
        )

        self.assertEqual(len(metadata.categorical_vocabularies), 11)
        self.assertNotIn(
            "DNS_QUERY_ID", metadata.categorical_vocabularies
        )
        self.assertEqual(
            metadata.training_sample_ids_hash,
            self.training_hash,
        )
        self.assertEqual(
            metadata.split_manifest_sha256,
            "split-manifest-sha256",
        )
        source_port = metadata.categorical_vocabularies[
            "L4_SRC_PORT"
        ]
        protocol = metadata.categorical_vocabularies["PROTOCOL"]
        source_index = self.val_sample.feature_ids.index(
            "L4_SRC_PORT"
        )
        protocol_index = self.val_sample.feature_ids.index(
            "PROTOCOL"
        )
        self.assertEqual(
            self.val_sample.features[source_index], 1
        )
        self.assertEqual(
            source_port.encode(
                self.val_sample.category_tokens[source_index]
            ),
            1277,
        )
        self.assertEqual(
            protocol.encode(
                self.val_sample.category_tokens[protocol_index]
            ),
            1,
        )
        self.assertEqual(
            protocol.id_to_token[2:],
            ("n:17", "n:6"),
        )
        self.assertEqual(protocol.known_frequencies, (1, 1))

    def test_validation_and_test_values_do_not_change_fitted_metadata(self):
        first = build_nf_unsw_tokenizer_metadata(
            self.dataset,
            self.preprocessor,
            data_manifest=self.data_manifest,
        )
        alternate_val = make_sample(
            self.preprocessor,
            raw_row(source_port="60000", protocol="777"),
            "sample-3",
        )
        alternate = NFUNSWDevelopmentSamples(
            splits={
                "train": self.train_samples,
                "val": (alternate_val,),
                "test": (),
            },
            preprocessor_hash=self.preprocessor.preprocessor_hash,
            split_manifest_sha256="split-manifest-sha256",
            raw_csv_sha1="raw-csv-sha1",
            schema_version=NF_UNSW_NB15_V2_SCHEMA.version,
            training_sample_ids_hash=self.training_hash,
        )

        second = build_nf_unsw_tokenizer_metadata(
            alternate,
            self.preprocessor,
            data_manifest=self.data_manifest,
        )

        self.assertEqual(first.metadata_hash, second.metadata_hash)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_rejects_preprocessor_not_bound_to_dataset(self):
        mismatched = NFUNSWDevelopmentSamples(
            splits=self.dataset.splits,
            preprocessor_hash="different-preprocessor",
            split_manifest_sha256="split-manifest-sha256",
            raw_csv_sha1="raw-csv-sha1",
            schema_version=NF_UNSW_NB15_V2_SCHEMA.version,
            training_sample_ids_hash=self.training_hash,
        )

        with self.assertRaisesRegex(
            ValueError, "preprocessor_hash"
        ):
            build_nf_unsw_tokenizer_metadata(
                mismatched,
                self.preprocessor,
                data_manifest=self.data_manifest,
            )


if __name__ == "__main__":
    unittest.main()
