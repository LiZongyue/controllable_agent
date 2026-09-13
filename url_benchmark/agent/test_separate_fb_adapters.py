# REDUNDANCY REVIEW: thin pytest wrapper that just re-runs
# scripts/check_separate_fb_adapters.py's check_agent_contract(), which already runs (and
# fails loudly) as the preflight gate inside launch_cheetah_separate_fb_adapters.sh and the
# launch_cheetah_sep_mlp_ln_*_routes_12.sh launchers before every real job. See also
# test_separate_fb_launcher.py, which re-verifies the same contract a third way via a full
# subprocess dry-run of the launcher.
from scripts.check_separate_fb_adapters import check_agent_contract


def test_separate_fb_adapter_contract() -> None:
    checks = check_agent_contract()
    assert checks
    assert all(checks.values())
