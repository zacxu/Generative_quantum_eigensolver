"""Confirmation authorization and provenance guards; no physical queries."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import section6_v2 as execution
from large_qubit.catalogue import build_screened_catalogue
from large_qubit.common import canonical_bytes, sha256_file
from large_qubit.tests.test_catalogue import synthetic_training
from large_qubit.tests.test_section6_v2 import SyntheticLedger


class ApprovedConfirmationTest(unittest.TestCase):
    def test_policy_retains_original_quality_and_separate_cumulative_budget(self):
        from large_qubit.acquisition import quality_policy
        catalogue, _ = build_screened_catalogue(synthetic_training())
        policy = execution.build_section6_policy(catalogue)
        self.assertEqual(policy['acquisition']['quality_policy'], quality_policy())
        self.assertEqual(policy['acquisition']['maximum_logical_requests'], 8192)
        self.assertEqual(policy['maximum_cumulative_ordinary_requests'], 6538 + 8192)
        self.assertEqual(policy['max_wall_seconds'], 86400)
        self.assertEqual(policy['authorized_confirmation_campaigns'], 1)
        self.assertTrue(policy['fresh_labels_only'])

    def test_unapproved_or_changed_amendment_stops_before_prerequisite_or_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / execution.AMENDMENT_PATH
            path.parent.mkdir(parents=True)
            path.write_text('{}', encoding='utf-8')
            with patch('large_qubit.section5.verify_section5', side_effect=AssertionError('No prior work before approval')) as prior:
                with self.assertRaisesRegex(ValueError, 'amendment changed'):
                    execution.prepare_section6(root)
                prior.assert_not_called()
            self.assertFalse((root / execution.PRIVATE_ROOT).exists())

    def test_hash_pin_does_not_replace_explicit_single_campaign_authorization(self):
        good = {'approved': True,
                'authorization': {'verbatim_user_answer': 'Approve one fresh v2 CPU campaign'},
                'stopping_and_scope': {'authorized_fresh_campaigns': 1, 'automatic_repeated_campaigns': False},
                'prospective_campaign': {'fresh_ordinary_request_ceiling': 8192,
                                         'maximum_cumulative_ordinary_requests': 14730}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / execution.AMENDMENT_PATH
            path.parent.mkdir(parents=True)
            for field in ('approved', 'answer', 'campaigns', 'repeat', 'fresh', 'cumulative'):
                value = deepcopy(good)
                if field == 'approved': value['approved'] = False
                elif field == 'answer': value['authorization']['verbatim_user_answer'] = 'maybe'
                elif field == 'campaigns': value['stopping_and_scope']['authorized_fresh_campaigns'] = 2
                elif field == 'repeat': value['stopping_and_scope']['automatic_repeated_campaigns'] = True
                elif field == 'fresh': value['prospective_campaign']['fresh_ordinary_request_ceiling'] = 9000
                else: value['prospective_campaign']['maximum_cumulative_ordinary_requests'] = 16000
                path.write_bytes(canonical_bytes(value))
                with self.subTest(field=field), patch.object(execution, 'AMENDMENT_SHA256', sha256_file(path)):
                    with self.assertRaisesRegex(ValueError, 'user authorization'):
                        execution._approved_amendment(root)

    def test_v1_row_identity_cannot_be_replayed_as_a_fresh_response(self):
        fixture = SyntheticLedger()
        for _ in range(3): fixture.query()
        started = fixture.append(fixture.start_payload())
        response = fixture.response_payload(started)
        self.assertIn('_v2_D', response['eligible_row']['row_id'])
        response['eligible_row']['row_id'] = response['eligible_row']['row_id'].replace('_v2_', '_')
        with self.assertRaisesRegex(ValueError, 'Retention'):
            fixture.replay([{'payload': response}], state=deepcopy(fixture.state))

    def test_cumulative_budget_cannot_treat_the_failed_attempt_as_free(self):
        fixture = SyntheticLedger()
        request = fixture.start_payload()
        with patch.object(execution, 'PRIOR_REQUESTS', execution.MAXIMUM_CUMULATIVE_REQUESTS):
            with self.assertRaisesRegex(ValueError, 'Request cap'):
                fixture.replay([{'payload': request}], state=deepcopy(fixture.state))
        self.assertEqual(fixture.state['issued'], 0)

    def test_preview_and_resume_without_execute_do_not_start_a_worker(self):
        with patch.object(execution, 'run_section6', side_effect=AssertionError('No implicit worker')) as run:
            with self.assertRaisesRegex(ValueError, 'Resume requires explicit'):
                execution.main(['acquire', '--resume'])
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
