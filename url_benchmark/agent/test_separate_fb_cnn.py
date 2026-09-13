# REDUNDANCY REVIEW: thin pytest wrapper that just re-runs
# scripts/check_separate_fb_cnn.py's check_agent_contract(), which already runs (and fails
# loudly) as the preflight gate inside launch_cheetah_separate_fb_cnn_onlineenc.sh before
# every real job. This test adds no coverage beyond re-invoking that same function.
from scripts.check_separate_fb_cnn import check_agent_contract


def test_separate_fb_cnn_contract() -> None:
    checks = check_agent_contract()
    assert checks
    assert all(checks.values())
