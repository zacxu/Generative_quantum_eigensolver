"""Synthetic, zero-Hamiltonian-call tests of the durable acquisition executor."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from large_qubit.common import canonical_bytes, digest, sha256_file
from large_qubit import section6_v2 as execution


ALIASES = execution.TRAINING
SOURCES = {'large_qubit/section6_v2.py': 'd' * 64}
CATALOGUE = {'sha256': 'c' * 64}
POLICY = {
    'acquisition': {'geometry_request_quotas': dict(zip(ALIASES, (2731, 2731, 2730)))},
    'memory_limit_bytes': 1024 ** 3, 'preflight_completed_requests': 18,
    'max_wall_seconds': 86400., 'cpu_threads': 4,
    'schedule_projection_retry_factor': 1.25,
}


def reservations():
    return {alias: [
        {'ops': [ordinal] + [1] * 31, 'diagnostic_ordinal': ordinal,
         'component': 'uniform' if ordinal % 2 else 'guided'}
        for ordinal in range(1, 129)] for alias in ALIASES}


class FakeSampler:
    """A replayable proposal source that exposes exactly which parents it sees."""

    def __init__(self, alias, catalogue=None, screening=None):
        self.alias = alias
        self.plans = {}
        self.parent_inputs = []

    def propose(self, ordinal, rows):
        self.parent_inputs.append(deepcopy(rows))
        if ordinal in self.plans:
            return deepcopy(self.plans[ordinal])
        parent = rows[-1]['row_id'] if rows else None
        return {'ops': [200 + ordinal] + [2] * 31,
                'fitting_request_ordinal': ordinal,
                'component': 'local' if parent else 'guided',
                'parent_row_id': parent, 'recipe': {'retain_if_novel': True}}


def norm():
    return {'state_norm': 1., 'norm_error': 0., 'n_alpha': 11, 'n_beta': 11,
            'sector_dimension': 1863225, 'sector_leakage_probability': 0.}


class SyntheticLedger:
    def __init__(self):
        self.policy, self.catalogue = deepcopy(POLICY), deepcopy(CATALOGUE)
        self.reservations = reservations()
        self.samplers = {alias: FakeSampler(alias) for alias in ALIASES}
        self.records, self.state = [], execution._fresh_state()

    def replay(self, records, *, state=None, samplers=None, **kwargs):
        return execution.replay_records(records, self.policy, self.catalogue, {}, self.reservations,
            state=state, samplers=self.samplers if samplers is None else samplers, **kwargs)

    def append(self, payload):
        record = {'payload': deepcopy(payload)}
        self.state = self.replay([record], state=self.state)
        self.records.append(record)
        return payload

    def start_payload(self):
        alias, role, plan = execution.next_request(
            self.state, self.samplers, self.reservations, self.policy)
        key = digest({'geometry_id': alias, 'ops': plan['ops'],
                      'catalogue_sha256': self.catalogue['sha256']})
        return {'event': 'request_started', 'request_ordinal': self.state['issued'] + 1,
                'geometry_request_ordinal': self.state['counts'][alias] + 1,
                'geometry_id': alias, 'role': role, 'proposal': plan,
                'catalogue_sha256': self.catalogue['sha256'], 'query_sha256': key,
                'duplicate_of': self.state['seen'].get(key), 'utc': '2026-11-05T00:00:00+00:00'}

    def response_payload(self, started, energy=-100.):
        alias, role, plan = (started['geometry_id'], started['role'], started['proposal'])
        ops = plan['ops']
        reserved = {tuple(value['ops']) for value in self.reservations[alias]}
        keep = role != 'HF' and (role == 'diagnostic' or
            (any(ops) and tuple(ops) not in reserved and tuple(ops) not in self.state['retained'][alias]))
        row = None
        if keep:
            number = plan['diagnostic_ordinal'] if role == 'diagnostic' else len(self.state['fitting'][alias]) + 1
            row = {'row_id': f'{alias}_v2_{"D" if role == "diagnostic" else "F"}{number:06d}',
                   'geometry_id': alias, 'role': role, 'ops': list(ops),
                   'energy_hartree': float(energy), 'catalogue_sha256': self.catalogue['sha256'],
                   'request_ordinal': started['request_ordinal']}
        result = {'event': 'request_completed', 'request_ordinal': started['request_ordinal'],
                  'geometry_id': alias, 'role': role, 'query_sha256': started['query_sha256'],
                  'energy_hartree': energy, 'eligible_row': row, 'norm': norm(),
                  'state_preparation_seconds': .01, 'wall_seconds': .02,
                  'peak_rss_bytes': 4096, 'hamiltonian_actions': 1}
        if role == 'HF':
            result['HF_energy_discrepancy_ha'] = 0.
        return result

    def query(self, energy=-100.):
        started = self.append(self.start_payload())
        self.append(self.response_payload(started, energy))
        return started

    def fitting_frontier(self):
        for _ in range(3 + 3 * 128):
            self.query()
        assert all(len(self.state['diagnostics'][alias]) == 128 for alias in ALIASES)
        return self


class Section6JournalTest(unittest.TestCase):
    def test_hash_chain_rejects_changed_payload_and_segment_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            journal = execution.Journal(folder)
            journal.append({'event': 'session_started', 'utc': 'before'})
            journal.append({'event': 'session_finished', 'utc': 'after', 'stop_reason': 'schedule'})
            journal.close()
            path = folder / 'journal_0001.jsonl'
            original = path.read_bytes()
            changed = original.replace(b'"utc":"before"', b'"utc":"forged"')
            self.assertNotEqual(original, changed)
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, 'hash chain'):
                execution.read_journal(folder)
            path.write_bytes(original)
            (folder / 'journal_0003.jsonl').write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'gap'):
                execution.read_journal(folder)

    def test_ambiguous_json_and_boolean_record_index_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'journal_0001.jsonl'
            body = {'record_index': 1, 'previous_sha256': execution.GENESIS,
                    'payload': {'event': 'session_started'}}
            value = {**body, 'sha256': digest(body)}
            duplicate = canonical_bytes(value).replace(b'"record_index":1', b'"record_index":99,"record_index":1')
            boolean_body = {**body, 'record_index': True}
            boolean = canonical_bytes({**boolean_body, 'sha256': digest(boolean_body)})
            for label, data in [('duplicate key', duplicate), ('boolean index', boolean)]:
                with self.subTest(label=label):
                    path.write_bytes(data + b'\n')
                    with self.assertRaises(ValueError):
                        execution.read_journal(temporary)

    def test_partial_tail_remains_byte_exact_when_resuming_in_next_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = execution.Journal(temporary)
            committed = first.append({'event': 'session_started', 'utc': 'first'})
            first.close()
            first_path = Path(temporary) / 'journal_0001.jsonl'
            fragment = b'{"record_index":2,"interrupted":'
            with first_path.open('ab') as handle:
                handle.write(fragment)
            original = first_path.read_bytes()
            resumed = execution.Journal(temporary)
            self.assertEqual(resumed.path.name, 'journal_0002.jsonl')
            self.assertEqual(resumed.previous, committed['sha256'])
            self.assertEqual(resumed.tails[0]['bytes'], len(fragment))
            resumed.append({'event': 'session_started', 'utc': 'resumed'})
            resumed.close()
            records, tails = execution.read_journal(temporary)
            self.assertEqual(first_path.read_bytes(), original)
            self.assertEqual([record['record_index'] for record in records], [1, 2])
            self.assertEqual(tails[0]['uncommitted_tail_sha256'], digest(list(fragment)))


class Section6ReplayTest(unittest.TestCase):
    def test_full_and_incremental_replay_are_identical_after_diagnostics_and_local_parent(self):
        fixture = SyntheticLedger().fitting_frontier()
        for _ in range(5):
            fixture.query(-100.125)
        full_samplers = {alias: FakeSampler(alias) for alias in ALIASES}
        full = fixture.replay(fixture.records, samplers=full_samplers)
        incremental = execution._fresh_state()
        incremental_samplers = {alias: FakeSampler(alias) for alias in ALIASES}
        for offset in range(0, len(fixture.records), 13):
            incremental = fixture.replay(fixture.records[offset:offset + 13], state=incremental,
                                         samplers=incremental_samplers)
        self.assertEqual(full, incremental)
        self.assertEqual(full, fixture.state)
        self.assertEqual(full['issued'], 392)
        self.assertEqual(len(full['eligible']), 389)
        for sampler in full_samplers.values():
            for parent_rows in sampler.parent_inputs:
                self.assertTrue(all(row['role'] == 'fitting' for row in parent_rows))

    def test_pending_failed_and_abandoned_intents_are_charged_and_retry_is_duplicate(self):
        fixture = SyntheticLedger()
        first = fixture.append(fixture.start_payload())
        self.assertEqual(fixture.state['issued'], 1)
        self.assertEqual(fixture.state['counts'][ALIASES[0]], 1)
        with self.assertRaisesRegex(ValueError, 'pending'):
            fixture.replay([{'payload': first}], state=deepcopy(fixture.state))
        fixture.append({'event': 'request_abandoned', 'request_ordinal': 1})
        second = fixture.append(fixture.start_payload())
        self.assertEqual(second['request_ordinal'], 2)
        self.assertEqual(second['geometry_request_ordinal'], 2)
        self.assertEqual(second['duplicate_of'], 1)
        fixture.append({'event': 'request_failed', 'request_ordinal': 2})
        self.assertEqual((fixture.state['issued'], fixture.state['completed'],
                          fixture.state['failed'], fixture.state['abandoned']), (2, 0, 1, 1))
        self.assertEqual(fixture.state['counts'][ALIASES[0]], 2)
        self.assertEqual(fixture.state['pending'], {})

    def test_collisions_duplicates_and_poor_local_rows_have_distinct_retention(self):
        fixture = SyntheticLedger().fitting_frontier()
        first = fixture.query(-99.)  # Deliberately worse than the -100 HF baseline.
        self.assertEqual(fixture.state['fitting'][ALIASES[0]][0]['energy_hartree'], -99.)
        for alias in ALIASES:
            fixture.samplers[alias].plans[1] = {
                'ops': list(fixture.reservations[alias][0]['ops']), 'component': 'reserved_collision'}
        # T02 and T03 collisions are queried but never relabelled as fitting data.
        for _ in range(2):
            collision = fixture.query(-100.)
            self.assertEqual(collision['role'], 'fitting')
            self.assertIsNotNone(collision['duplicate_of'])
        self.assertEqual(sum(map(len, fixture.state['fitting'].values())), 1)
        # Balanced frontier returns to T02, whose first proposal retained nothing.
        alias = execution.next_request(fixture.state, fixture.samplers, fixture.reservations, fixture.policy)[0]
        fixture.samplers[alias].plans[2] = {'ops': list(fixture.reservations[alias][0]['ops']),
                                          'component': 'reserved_collision'}
        repeated = fixture.append(fixture.start_payload())
        changed_energy = fixture.response_payload(repeated, -100.01)
        with self.assertRaisesRegex(ValueError, 'inconsistent energies'):
            fixture.replay([{'payload': changed_energy}], state=deepcopy(fixture.state))
        fixture.append(fixture.response_payload(repeated, -100.))
        self.assertEqual(fixture.state['issued'], 391)
        self.assertEqual(fixture.state['fit_requests'][alias], 2)
        self.assertEqual(len(fixture.state['fitting'][alias]), 0)
        self.assertEqual(fixture.state['fitting'][ALIASES[0]][0]['ops'], first['proposal']['ops'])
        # Give the other geometries their first novel fitting rows, then repeat
        # T01's already retained fitting circuit (distinct from a reservation).
        for _ in range(2):
            fixture.query(-99.)
        fixture.samplers[ALIASES[0]].plans[2] = {
            'ops': list(first['proposal']['ops']), 'component': 'retained_duplicate'}
        repeated = fixture.query(-99.)
        self.assertEqual((repeated['geometry_id'], repeated['duplicate_of']), (ALIASES[0], 388))
        self.assertEqual(sum(map(len, fixture.state['fitting'].values())), 3)
        self.assertEqual(fixture.state['fit_requests'][ALIASES[0]], 2)
        # A later local child is retained even though it worsens its parent.
        local = fixture.query(-98.)
        self.assertEqual(local['proposal']['component'], 'local')
        rows = fixture.state['fitting'][local['geometry_id']]
        self.assertEqual((len(rows), rows[-1]['energy_hartree']), (2, -98.))

    def test_altered_proposal_parent_source_response_or_norm_is_rejected(self):
        fixture = SyntheticLedger().fitting_frontier()
        for _ in range(3):
            fixture.query()
        original = fixture.start_payload()
        self.assertIsNotNone(original['proposal']['parent_row_id'])
        changes = [
            ('ops', lambda item: item['proposal']['ops'].__setitem__(0, 400)),
            ('parent', lambda item: item['proposal'].__setitem__('parent_row_id', ALIASES[0] + '_v2_D000001')),
            ('component', lambda item: item['proposal'].__setitem__('component', 'uniform')),
            ('catalogue', lambda item: item.__setitem__('catalogue_sha256', '0' * 64)),
            ('duplicate', lambda item: item.__setitem__('duplicate_of', 1)),
        ]
        for label, change in changes:
            altered = deepcopy(original); change(altered)
            with self.subTest(label=label), self.assertRaises(ValueError):
                fixture.replay([{'payload': altered}], state=deepcopy(fixture.state))
        session = {'event': 'session_started', 'utc': '2026-11-05T00:00:00+00:00',
                   'source_bindings': {'large_qubit/section6_v2.py': 'f' * 64}}
        with self.assertRaisesRegex(ValueError, 'source'):
            fixture.replay([{'payload': session}], expected_sources=SOURCES)
        fixture.append(original)
        good = fixture.response_payload(original)
        for field, value in [('state_norm', 1.01), ('norm_error', -1.),
                             ('n_alpha', 10), ('sector_leakage_probability', .01)]:
            altered = deepcopy(good); altered['norm'][field] = value
            with self.subTest(norm=field), self.assertRaises(ValueError):
                fixture.replay([{'payload': altered}], state=deepcopy(fixture.state))
        for field, value in [('hamiltonian_actions', True), ('peak_rss_bytes', -1),
                             ('energy_hartree', float('inf')), ('query_sha256', 'f' * 64)]:
            altered = deepcopy(good); altered[field] = value
            with self.subTest(response=field), self.assertRaises(ValueError):
                fixture.replay([{'payload': altered}], state=deepcopy(fixture.state))

    def test_budget_frontier_enforces_global_and_geometry_caps_without_reset(self):
        fixture = SyntheticLedger()
        state = deepcopy(fixture.state)
        state['counts'][ALIASES[0]] = POLICY['acquisition']['geometry_request_quotas'][ALIASES[0]]
        self.assertEqual(execution.next_request(state, fixture.samplers, fixture.reservations, fixture.policy)[0],
                         ALIASES[1])
        state['counts'] = deepcopy(POLICY['acquisition']['geometry_request_quotas'])
        self.assertIsNone(execution.next_request(state, fixture.samplers, fixture.reservations, fixture.policy))
        state = deepcopy(fixture.state); state['issued'] = 8192
        started = fixture.start_payload(); started['request_ordinal'] = 8193
        with self.assertRaisesRegex(ValueError, 'cap'):
            fixture.replay([{'payload': started}], state=state)
        state = deepcopy(fixture.state)
        state['counts'][ALIASES[0]] = POLICY['acquisition']['geometry_request_quotas'][ALIASES[0]]
        with self.assertRaisesRegex(ValueError, 'frontier'):
            fixture.replay([{'payload': fixture.start_payload()}], state=state)
        # A schedule-based fallback must use measured cost and the real window,
        # rather than an editable projection that can manufacture exhaustion.
        measured = SyntheticLedger()
        measured.append({'event': 'session_started', 'utc': '2026-11-05T00:00:00+00:00'})
        for _ in range(18):
            measured.query()
        mean = measured.state['completed_wall_seconds'] / measured.state['completed']
        preflight = {'event': 'preflight', 'utc': '2026-11-05T01:00:00+00:00',
                     'seconds_per_request': mean,
                     'projected_remaining_seconds_with_margin': mean * (8192 - 18) * 1.25,
                     'remaining_wall_seconds': 23 * 3600.}
        for field in ('seconds_per_request', 'projected_remaining_seconds_with_margin', 'remaining_wall_seconds'):
            changed = deepcopy(preflight); changed[field] += 1000.
            with self.subTest(preflight=field), self.assertRaisesRegex(ValueError, 'preflight'):
                measured.replay([{'payload': changed}], state=deepcopy(measured.state))
        measured.append(preflight)
        self.assertEqual(measured.state['preflight'], preflight)

    def test_all_diagnostics_precede_fitting_and_never_supply_parents(self):
        fixture = SyntheticLedger()
        for _ in range(3):
            self.assertEqual(fixture.query()['role'], 'HF')
        for ordinal in range(1, 129):
            for alias in ALIASES:
                started = fixture.query()
                self.assertEqual((started['geometry_id'], started['role'],
                                  started['proposal']['diagnostic_ordinal']), (alias, 'diagnostic', ordinal))
                self.assertTrue(all(not sampler.parent_inputs for sampler in fixture.samplers.values()))
        self.assertEqual(sum(map(len, fixture.state['diagnostics'].values())), 384)
        for _ in range(4):
            self.assertEqual(fixture.query()['role'], 'fitting')
        for sampler in fixture.samplers.values():
            for rows in sampler.parent_inputs:
                self.assertTrue(all(row['role'] == 'fitting' and '_F' in row['row_id'] for row in rows))
        started = next(record['payload'] for record in fixture.records
                       if record['payload']['event'] == 'request_started' and record['payload']['role'] == 'diagnostic')
        altered = deepcopy(started); altered['role'] = 'fitting'
        initial = fixture.replay(fixture.records[:6])
        with self.assertRaisesRegex(ValueError, 'frontier'):
            fixture.replay([{'payload': altered}], state=initial)

    def test_integrity_abandonment_is_separate_and_invalid_norm_or_error_is_rejected(self):
        fixture = SyntheticLedger()
        for _ in range(4):
            fixture.query()
        row = fixture.state['eligible'][0]
        ordinary = fixture.state['issued']
        intent = {'event': 'integrity_started', 'integrity_ordinal': 1,
                  'row_id': row['row_id'], 'request_ordinal': row['request_ordinal'],
                  'geometry_id': row['geometry_id']}
        fixture.append(intent)
        with self.assertRaisesRegex(ValueError, 'pending'):
            fixture.replay([{'payload': fixture.start_payload()}], state=deepcopy(fixture.state))
        fixture.append({'event': 'integrity_abandoned', 'integrity_ordinal': 1})
        self.assertEqual(fixture.state['integrity_issued'], 1)
        self.assertFalse(fixture.state['integrity_pending'])
        fixture.append({**intent, 'integrity_ordinal': 2})
        good = {'event': 'integrity_completed', 'integrity_ordinal': 2,
                'row_id': row['row_id'], 'request_ordinal': row['request_ordinal'],
                'geometry_id': row['geometry_id'], 'absolute_energy_discrepancy_ha': 0.,
                'norm': norm(), 'wall_seconds': .02}
        for field, value in [('absolute_energy_discrepancy_ha', -1.),
                             ('absolute_energy_discrepancy_ha', float('inf')),
                             ('geometry_id', ALIASES[1]), ('request_ordinal', row['request_ordinal'] + 1)]:
            changed = deepcopy(good); changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                fixture.replay([{'payload': changed}], state=deepcopy(fixture.state))
        changed = deepcopy(good); changed['norm']['state_norm'] = .9
        with self.assertRaises(ValueError):
            fixture.replay([{'payload': changed}], state=deepcopy(fixture.state))
        fixture.append(good)
        self.assertEqual(fixture.state['issued'], ordinary)
        self.assertEqual(fixture.state['completed'], ordinary)
        self.assertEqual(fixture.state['integrity_issued'], 2)
        self.assertEqual(len(fixture.state['integrity_checks']), 1)
        self.assertAlmostEqual(fixture.state['integrity_wall_seconds'], .02)

    def test_hf_scalar_and_exact_retained_row_are_bound_to_response(self):
        fixture = SyntheticLedger()
        started = fixture.append(fixture.start_payload())
        response = fixture.response_payload(started)
        with self.assertRaisesRegex(ValueError, 'authenticated SCF'):
            fixture.replay([{'payload': response}], state=deepcopy(fixture.state),
                           expected_hf={alias: -101. for alias in ALIASES})
        fixture.append(response)
        for _ in range(3):
            fixture.query()
        started = fixture.append(fixture.start_payload())
        response = fixture.response_payload(started)
        for change in ('drop', 'ordinal', 'ops', 'energy'):
            altered = deepcopy(response)
            if change == 'drop':
                altered['eligible_row'] = None
            elif change == 'ordinal':
                altered['eligible_row']['request_ordinal'] = 999
            elif change == 'ops':
                altered['eligible_row']['ops'][0] = 512
            else:
                altered['eligible_row']['energy_hartree'] -= .1
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Retention'):
                fixture.replay([{'payload': altered}], state=deepcopy(fixture.state))

    def test_keyboard_interrupt_preserves_intent_and_explicit_resume_charges_abandonment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / execution.PRIVATE_ROOT
            private.mkdir(parents=True)
            fixture = SyntheticLedger()
            prepared = (fixture.policy, fixture.catalogue, {}, fixture.reservations, {})
            locking = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=4, flock=lambda *args: None)
            basis = SimpleNamespace(hf_state=lambda: object(),
                apply_circuit=lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt('synthetic stop')))
            evaluator = SimpleNamespace(basis=basis)
            metadata = {'scf': {'total_energy_ha': -100.}}
            common = [patch.dict(sys.modules, {'fcntl': locking}),
                      patch.object(execution, 'verify_preparation', return_value=prepared),
                      patch.object(execution, '_sources', return_value=SOURCES),
                      patch.object(execution, '_expected_hf', return_value={a: -100. for a in ALIASES}),
                      patch('large_qubit.acquisition_v2.AcquisitionSampler', FakeSampler),
                      patch('large_qubit.catalogue.circuit_gates', return_value=[])]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for context in common:
                    stack.enter_context(context)
                loader = stack.enter_context(patch('large_qubit.section4.load_co2_evaluator',
                                                    return_value=(evaluator, metadata)))
                with self.assertRaises(KeyboardInterrupt):
                    execution.run_section6(root)
                first_path = private / 'journal_0001.jsonl'
                first_bytes = first_path.read_bytes()
                first_records, _ = execution.read_journal(private)
                before = fixture.replay(first_records, expected_sources=SOURCES)
                self.assertEqual((before['issued'], len(before['pending'])), (1, 1))
                self.assertFalse((root / execution.DEFAULT_SECTION6 / 'manifest.json').exists())
                loader.side_effect = KeyboardInterrupt('stop after resume abandonment')
                with self.assertRaises(KeyboardInterrupt):
                    execution.run_section6(root, resume=True)
            records, _ = execution.read_journal(private)
            after = fixture.replay(records, expected_sources=SOURCES)
            self.assertEqual(first_path.read_bytes(), first_bytes)
            self.assertTrue((private / 'journal_0002.jsonl').is_file())
            self.assertEqual((after['issued'], after['abandoned'], after['completed']), (1, 1, 0))
            self.assertEqual(after['counts'][ALIASES[0]], 1)
            self.assertFalse(after['pending'])
            errors = list((root / execution.DEFAULT_SECTION6 / 'attempt_errors').glob('*.json'))
            self.assertEqual(len(errors), 2)
            self.assertTrue(all(json.loads(path.read_text())['partial_evidence_preserved'] for path in errors))


class Section6RecoveryTest(unittest.TestCase):
    def test_commit_is_atomic_idempotent_no_clobber_and_keeps_every_staged_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            target, data = folder / 'manifest.json', b'{"complete":true}\n'
            with patch.object(execution.os, 'link', side_effect=OSError('synthetic interrupted commit')):
                with self.assertRaises(OSError):
                    execution._commit_bytes(target, data)
            self.assertFalse(target.exists())
            candidates = list((folder / 'commit_attempts').glob('*.pending'))
            self.assertEqual([candidate.read_bytes() for candidate in candidates], [data])
            native_link = execution.os.link

            def publish_complete_bytes(candidate, destination):
                self.assertEqual(Path(candidate).read_bytes(), data)
                self.assertFalse(Path(destination).exists())
                native_link(candidate, destination)

            with patch.object(execution.os, 'link', side_effect=publish_complete_bytes) as link:
                execution._commit_bytes(target, data)
                link.assert_called_once()
            self.assertEqual(target.read_bytes(), data)
            before = {path.name: path.read_bytes() for path in (folder / 'commit_attempts').iterdir()}
            self.assertEqual(len(before), 2)
            execution._commit_bytes(target, data)
            self.assertEqual({path.name: path.read_bytes() for path in (folder / 'commit_attempts').iterdir()}, before)
            with self.assertRaisesRegex(ValueError, 'differs'):
                execution._commit_bytes(target, b'changed')
            self.assertEqual(target.read_bytes(), data)
            # A concurrent publisher also retains its artifact and our staged
            # bytes; even the race path never replaces an existing destination.
            raced = folder / 'raced.json'

            def concurrent_publish(candidate, destination):
                Path(destination).write_bytes(b'other frozen evidence')
                raise FileExistsError(str(destination))

            with patch.object(execution.os, 'link', side_effect=concurrent_publish):
                with self.assertRaisesRegex(ValueError, 'Concurrent'):
                    execution._commit_bytes(raced, data)
            self.assertEqual(raced.read_bytes(), b'other frozen evidence')
            self.assertEqual(len(list((folder / 'commit_attempts').iterdir())), 3)
            self.assertTrue(all(candidate.read_bytes() == data for candidate in (folder / 'commit_attempts').iterdir()))

    def test_finish_frozen_recovers_missing_sidecar_and_refuses_mismatched_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = root / execution.PRIVATE_ROOT, root / execution.DEFAULT_SECTION6
            private.mkdir(parents=True); public.mkdir(parents=True)
            raw = private / 'journal_0001.jsonl'
            raw.write_bytes(b'original immutable journal\n')
            prep = root / execution.PREPARATION_PATH
            prep.parent.mkdir(parents=True)
            prep.write_bytes(b'original preparation\n')
            inputs = {'source_bindings': SOURCES, 'preparation_file_sha256': sha256_file(prep),
                      'manifest_base': {'files': {raw.relative_to(root).as_posix(): sha256_file(raw)},
                                        'release': None, 'section_6_complete': False,
                                        'feasibility_gate': 'pending_finalization'},
                      'release_arguments': None}
            frozen_manifest = {**inputs['manifest_base'], 'release': None,
                               'section_6_complete': False, 'feasibility_gate': 'no_go'}
            manifest = public / 'manifest.json'
            manifest.write_bytes(canonical_bytes(frozen_manifest) + b'\n')
            original = manifest.read_bytes()
            sentinel = {'all_passed': True, 'section_6_complete': False}
            with patch.object(execution, '_sources', return_value=SOURCES), \
                    patch.object(execution, 'verify_section6', return_value=sentinel) as verify, \
                    patch('large_qubit.corpus_v2.build_release', side_effect=AssertionError('no release requested')):
                self.assertEqual(execution._finish_frozen(root, public, deepcopy(inputs)), sentinel)
                self.assertEqual(manifest.read_bytes(), original)
                sidecar = public / 'manifest.sha256'
                self.assertEqual(sidecar.read_text().strip(), sha256_file(manifest))
                self.assertEqual(execution._finish_frozen(root, public, deepcopy(inputs)), sentinel)
                self.assertEqual(verify.call_count, 2)
                for field in ('source_bindings', 'preparation_file_sha256'):
                    altered = deepcopy(inputs)
                    altered[field] = {} if field == 'source_bindings' else 'f' * 64
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Frozen finalization'):
                        execution._finish_frozen(root, public, altered)
                raw.write_bytes(b'changed raw journal\n')
                with self.assertRaisesRegex(ValueError, 'raw evidence changed'):
                    execution._finish_frozen(root, public, deepcopy(inputs))
                self.assertEqual(manifest.read_bytes(), original)
                raw.write_bytes(b'original immutable journal\n')
                manifest.write_bytes(b'conflicting frozen manifest\n')
                with self.assertRaisesRegex(ValueError, 'committed artifact differs'):
                    execution._finish_frozen(root, public, deepcopy(inputs))
                self.assertEqual(manifest.read_bytes(), b'conflicting frozen manifest\n')
                self.assertEqual(verify.call_count, 2)

    def test_finalization_resume_finishes_before_journal_or_evaluator_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private, public = root / execution.PRIVATE_ROOT, root / execution.DEFAULT_SECTION6
            private.mkdir(parents=True); public.mkdir(parents=True)
            inputs = {'frozen': 'finalization arguments'}
            (public / 'finalization_input.json').write_bytes(canonical_bytes(inputs) + b'\n')
            # This also models the crash after the manifest was published but
            # before its sidecar and the final verification were completed.
            (public / 'manifest.json').write_bytes(b'existing matching manifest\n')
            fixture, sentinel = SyntheticLedger(), {'all_passed': True}
            prepared = (fixture.policy, fixture.catalogue, {}, fixture.reservations, {})
            locking = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=4,
                                      flock=unittest.mock.Mock())
            with patch.dict(sys.modules, {'fcntl': locking}), \
                    patch.object(execution, 'verify_preparation', return_value=prepared), \
                    patch.object(execution, '_finish_frozen', return_value=sentinel) as finish, \
                    patch.object(execution, 'Journal', side_effect=AssertionError('must not construct journal')) as journal, \
                    patch('large_qubit.section4.load_co2_evaluator',
                          side_effect=AssertionError('must not load evaluator')) as evaluator, \
                    patch.object(execution, '_expected_hf', side_effect=AssertionError('must not load HF')):
                self.assertEqual(execution.run_section6(root, resume=True), sentinel)
                finish.assert_called_once_with(root.resolve(), public.resolve(), inputs)
                journal.assert_not_called()
                evaluator.assert_not_called()
            self.assertEqual(locking.flock.call_count, 2)
            self.assertFalse(list(private.glob('journal_*.jsonl')))
            self.assertEqual((public / 'manifest.json').read_bytes(), b'existing matching manifest\n')

    def test_persisted_terminal_stop_or_failed_preflight_cannot_issue_more_requests(self):
        for reason in ('target', 'budget', 'schedule'):
            fixture = SyntheticLedger()
            fixture.append({'event': 'session_finished', 'stop_reason': reason,
                            'utc': '2026-11-05T00:00:01+00:00'})
            fixture.append({'event': 'session_started', 'resume': True,
                            'utc': '2026-11-05T00:00:02+00:00'})
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, 'terminal'):
                fixture.replay([{'payload': fixture.start_payload()}], state=deepcopy(fixture.state))
            self.assertEqual(fixture.state['issued'], 0)
        measured = SyntheticLedger()
        measured.append({'event': 'session_started', 'utc': '2026-11-05T00:00:00+00:00'})
        for _ in range(18):
            measured.query()
        mean = measured.state['completed_wall_seconds'] / measured.state['completed']
        measured.append({'event': 'preflight', 'utc': '2026-11-05T23:59:50+00:00',
                         'seconds_per_request': mean,
                         'projected_remaining_seconds_with_margin': mean * (8192 - 18) * 1.25,
                         'remaining_wall_seconds': 10.})
        measured.append({'event': 'session_started', 'resume': True,
                         'utc': '2026-11-05T23:59:51+00:00'})
        self.assertIsNone(measured.state['last_stop_reason'])
        with self.assertRaisesRegex(ValueError, 'terminal'):
            measured.replay([{'payload': measured.start_payload()}], state=deepcopy(measured.state))
        self.assertEqual(measured.state['issued'], 18)


if __name__ == '__main__':
    unittest.main()
