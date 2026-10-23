"""A scientific-library process stays local unless its caller installs an executor."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest

from organelleverse.assembly.environment_contracts import EnvironmentResolution
from organelleverse.assembly.environment_resolver import EnvironmentResolver, require_local_resolution
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.operations.execution_hooks import (
    ExecutionDelegated, ExecutionPolicy, ExecutionRequirement, execution_scope,
    local_execution_required, select_execution, set_process_execution_policy,
    staged_input_copy_required,
)

class Hooks:
    def __init__(self):self.calls=[]
    def select(self, requirement):
        self.calls.append(requirement)
        return {'caller-owned':'selection'}

def test_no_installed_hook_means_no_target_discovery():
    with execution_scope(hooks=None):
        assert select_execution(ExecutionRequirement(reason='missing_dependencies')) is None

def test_local_scope_isolation_and_restore():
    hooks=Hooks()
    with execution_scope(hooks=hooks):
        with execution_scope(local_only=True,copy_staged_inputs=True):
            assert select_execution(ExecutionRequirement(reason='missing_dependencies')) is None
            assert staged_input_copy_required()
        assert select_execution(ExecutionRequirement(reason='missing_dependencies'))=={'caller-owned':'selection'}
    assert len(hooks.calls)==1

def test_dedicated_worker_policy_crosses_native_thread_boundary():
    previous=set_process_execution_policy(ExecutionPolicy(local_only=True,copy_staged_inputs=True))
    try:
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(lambda:(local_execution_required(),staged_input_copy_required())).result()==(True,True)
    finally:
        set_process_execution_policy(previous)

def test_historic_external_record_decodes_without_application_models():
    selection={'target_id':'ssh:old','provider_id':'ssh','transport_kind':'ssh_mcp',
               'platform':'linux-64','architecture':'x86_64','artifact_transport':'ssh_content_cache'}
    record={'selected_provider':None,'managed_plan':None,'compute_target':selection,'rejections':[]}
    resolution=EnvironmentResolution.model_validate(record)
    assert resolution.model_dump(mode='json')==record
    with pytest.raises(ExecutionDelegated) as caught:
        require_local_resolution(resolution,backend_id='oatk')
    assert caught.value.selection==selection

def test_library_windows_error_has_no_app_installation_side_effect():
    with execution_scope(hooks=None),pytest.raises(OrganelleDependencyError) as caught:
        EnvironmentResolver(_host_os='nt')._resolve_on_windows('oatk',SimpleNamespace(environment_source='auto'))
    assert caught.value.details['needs']=='linux_environment'
    assert not caught.value.suggested_action
